
class AnalysisException(Exception):

    def __init__(self, message):
        self.message = message

    def what(self):
        return self.message
