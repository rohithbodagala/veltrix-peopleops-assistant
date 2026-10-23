"""Fixed user-facing texts. Refusals never echo names or say whether an employee or a
restricted document exists."""

REFUSE_EMPLOYEE_DATA = "I can't share information about that employee."
REFUSE_BULK = (
    "I can't share leave information about many employees at once. "
    "I can help with your own leave, or with one employee you are allowed to see."
)
REFUSE_TEAM_NOT_MANAGER = "Team leave information is only available to managers, for their own direct reports."
REFUSE_ACT_FOR_OTHERS = "I can only request or cancel leave for you, not on behalf of other employees."
REFUSE_CONTACT = "I can't share personal contact details such as phone numbers or email addresses."
REFUSE_HALF_DAY = (
    "Leave can only be taken in whole days, so I can't request a half day. "
    "If you'd like a full day instead, tell me the date and whether it is annual or sick leave."
)
REFUSE_MANIPULATION = (
    "I can't help with that. I can answer questions about Veltrix HR policies and public holidays, "
    "and help you check, request or cancel your own leave."
)
REFUSE_OUT_OF_SCOPE = (
    "Sorry, that's outside what I can help with. I can answer questions about Veltrix HR policies and "
    "public holidays, and help you check, request or cancel your leave."
)
GREETING = (
    "Hello! I can answer questions about Veltrix HR policies and public holidays, show leave balances and "
    "requests you are allowed to see, and help you request or cancel your own leave. What would you like to do?"
)
NO_ANSWER = (
    "The HR documents don't cover this question, so I can't give you a reliable answer. "
    "Please contact the HR team for help."
)
LLM_UNAVAILABLE = (
    "I can't answer that right now because the assistant's language service is unavailable. "
    "Please try again shortly, or contact the HR team."
)
PROPOSAL_DISCARDED = "Okay, I won't go ahead with that. Is there anything else you'd like to change or ask?"
