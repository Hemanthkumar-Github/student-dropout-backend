questions = [
    "What is the student’s attendance percentage?",
    "What are the student’s average marks?",
    "What is the student’s family income level (1–5)?",
    "How many hours does the student study daily?",
    "How many subjects has the student failed?"

]

sessions = {}

def get_next_question(session_id, user_input=None):
    if session_id not in sessions:
        sessions[session_id] = {
            "step": -1,
            "data": [],
            "started": False
        }
        return "Hello! I’m your Student Dropout Risk Assistant. May I know your student name?"

    session = sessions[session_id]

    # First user message (name / greeting)
    if not session["started"]:
        session["started"] = True
        session["step"] = 0
        return questions[0]

    # Now numeric answers only
    try:
        session["data"].append(float(user_input))
        session["step"] += 1
    except:
        return "Please enter a numeric value."

    if session["step"] < len(questions):
        return questions[session["step"]]

    return "DONE"
