


class SessionState:
    def __init__(self):
        self.chunks_seen = 0
        self.total_duration = 0.0
        self.risk_scores = []
        self.repeated_phrases = {}
        self.urgency_score = 0
        self.politeness_score = 0
        self.last_verdict = None
