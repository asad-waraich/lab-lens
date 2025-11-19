# model-development/scripts/biobert_extractive_summarizer.py

class MedicalReportSummarizer:
    def __init__(self):
        # Load BioBERT
        self.tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        self.model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
        
    def summarize(self, row):
        """
        Uses your features + BioBERT to create summary
        """
        text = row['cleaned_text_final']
        sentences = text.split('. ')
        
        # Score sentences using your features
        sentence_scores = []
        for sent in sentences:
            score = self._calculate_importance(sent, row)
            sentence_scores.append(score)
        
        # Get BioBERT embeddings for semantic importance
        biobert_scores = self._get_biobert_scores(sentences)
        
        # Combine scores (your features + BioBERT)
        final_scores = [
            0.6 * feat_score + 0.4 * bert_score 
            for feat_score, bert_score in zip(sentence_scores, biobert_scores)
        ]
        
        # Select top sentences
        top_indices = np.argsort(final_scores)[-5:]  # Top 5 sentences
        summary = '. '.join([sentences[i] for i in sorted(top_indices)])
        
        return summary
    
    def _calculate_importance(self, sentence, features):
        """Use your 47 features to score sentence importance"""
        score = 0
        
        # Urgent cases - prioritize severity indicators
        if features['urgency_indicator'] == 1:
            urgency_words = ['urgent', 'critical', 'severe', 'immediate']
            score += sum(2 for word in urgency_words if word in sentence.lower())
        
        # Abnormal findings
        if features['abnormal_lab_ratio'] > 0.3:
            abnormal_words = ['abnormal', 'elevated', 'high', 'low']
            score += sum(1.5 for word in abnormal_words if word in sentence.lower())
        
        # Medication mentions (important for discharge)
        med_score = features['kw_medications'] / 10  # Normalize
        if any(med in sentence for med in ['mg', 'medication', 'dose']):
            score += med_score
            
        return score