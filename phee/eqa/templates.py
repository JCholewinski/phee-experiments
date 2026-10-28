"""Question templates for extractive QA (and stage 2 of generative QA).

Three variants (work plan, section "Extractive QA"):

  authors      templates of the PHEE repository configs (EEQA):
                 main arguments  "<argument query> in <trigger>?"    (stage 1, nth_query 5)
                                 e.g. "What is the treatment in induced?"
                 sub-arguments   "<Argument.Name> in <event type>"   (stage 2, nth_query 1)
                                 e.g. "Treatment.Drug in adverse event"
                 trigger         the authors detect triggers by token classification; in our
                                 QA formulation we ask "Trigger in <event type>" in their style
  current      the questions of the existing implementation (legacy/src/preprocessing)
  definitions  the current questions followed by the definition of the label
"""

from phee.schema import AUTHORS_LABEL_NAME, DEFINITIONS, EVENT_TYPE_NAME, TRIGGER

VARIANTS = ("authors", "current", "definitions")

_AUTHORS_QUERY = {
    "SUBJECT": "Who is the subject?",
    "TREATMENT": "What is the treatment?",
    "EFFECT": "What is the effect?",
}

_CURRENT_TRIGGER = {
    "ADE": "What phrase indicates an adverse drug event?",
    "PTE": "What phrase indicates a potential therapeutic event?",
}

_CURRENT_MAIN = {
    "SUBJECT": 'Who or what is involved in the event triggered by "{trigger}"?',
    "TREATMENT": 'What treatment is involved in the event triggered by "{trigger}"?',
    "EFFECT": 'What effect is described in the event triggered by "{trigger}"?',
}

_CURRENT_SUB = {
    "SUBJECT_AGE": 'What is the age of the subject "{main}"?',
    "SUBJECT_GENDER": 'What is the gender of the subject "{main}"?',
    "SUBJECT_RACE": 'What is the race of the subject "{main}"?',
    "SUBJECT_POPULATION": 'What population describes the subject "{main}"?',
    "SUBJECT_DISORDER": 'What disorder describes the subject "{main}"?',
    "TREATMENT_DRUG": 'What drug is mentioned in the treatment "{main}"?',
    "TREATMENT_DOSAGE": 'What dosage is mentioned for the treatment "{main}"?',
    "TREATMENT_ROUTE": 'What route is mentioned for the treatment "{main}"?',
    "TREATMENT_FREQ": 'What frequency is mentioned for the treatment "{main}"?',
    "TREATMENT_DURATION": 'What duration is mentioned for the treatment "{main}"?',
    "TREATMENT_TIME_ELAPSED": 'What time elapsed is mentioned for the treatment "{main}"?',
    "TREATMENT_DISORDER": 'What disorder is treated by "{main}"?',
}


def trigger_question(event_type, variant):
    if variant == "authors":
        return f"Trigger in {EVENT_TYPE_NAME[event_type]}"
    question = _CURRENT_TRIGGER[event_type]
    if variant == "definitions":
        question += f" {DEFINITIONS[event_type]} {DEFINITIONS[TRIGGER]}"
    return question


def main_question(label, trigger_text, event_type, variant):
    if variant == "authors":
        return f"{_AUTHORS_QUERY[label][:-1]} in {trigger_text}?"
    question = _CURRENT_MAIN[label].format(trigger=trigger_text)
    if variant == "definitions":
        question += f" {DEFINITIONS[label]}"
    return question


def sub_question(label, main_text, event_type, variant):
    if variant == "authors":
        return f"{AUTHORS_LABEL_NAME[label]} in {EVENT_TYPE_NAME.get(event_type, 'event')}"
    question = _CURRENT_SUB[label].format(main=main_text)
    if variant == "definitions":
        question += f" {DEFINITIONS[label]}"
    return question
