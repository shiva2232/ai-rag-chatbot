from plyer import notification

def notify_user(question, answer):
    notification.notify(
        title="⚠️ RAG Agent Needs Help",
        message=f"Question: {question[:100]}...\nAnswer: {answer[:100]}",
        timeout=10  # seconds
    )