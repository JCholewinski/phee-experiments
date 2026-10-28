"""Label inventory used by every method and by the evaluation.

Decisions recorded in the work plan:
  * Combination.Drug is removed everywhere (every such span is also a Treatment.Drug span).
  * Event attributes (negation, speculation, severity) are not modelled.
"""

EVENT_TYPES = ["ADE", "PTE"]

SOURCE_EVENT_TYPE = {
    "Adverse_event": "ADE",
    "Potential_therapeutic_event": "PTE",
}

EVENT_TYPE_NAME = {
    "ADE": "adverse event",
    "PTE": "potential therapeutic event",
}

TRIGGER = "TRIGGER"

MAIN_LABELS = ["SUBJECT", "TREATMENT", "EFFECT"]

SOURCE_MAIN_ROLE = {
    "Subject": "SUBJECT",
    "Treatment": "TREATMENT",
    "Effect": "EFFECT",
}

# Sub-arguments, grouped by their parent main argument.
SUB_LABELS_BY_PARENT = {
    "SUBJECT": [
        "SUBJECT_AGE",
        "SUBJECT_GENDER",
        "SUBJECT_RACE",
        "SUBJECT_POPULATION",
        "SUBJECT_DISORDER",
    ],
    "TREATMENT": [
        "TREATMENT_DRUG",
        "TREATMENT_DOSAGE",
        "TREATMENT_ROUTE",
        "TREATMENT_FREQ",
        "TREATMENT_DURATION",
        "TREATMENT_TIME_ELAPSED",
        "TREATMENT_DISORDER",
    ],
}

SUB_LABELS = [label for labels in SUB_LABELS_BY_PARENT.values() for label in labels]

SUB_PARENT = {
    label: parent
    for parent, labels in SUB_LABELS_BY_PARENT.items()
    for label in labels
}

# (main role in the source data, sub key in the source data) -> our label
SOURCE_SUB_ROLE = {
    ("Subject", "Age"): "SUBJECT_AGE",
    ("Subject", "Gender"): "SUBJECT_GENDER",
    ("Subject", "Race"): "SUBJECT_RACE",
    ("Subject", "Population"): "SUBJECT_POPULATION",
    ("Subject", "Disorder"): "SUBJECT_DISORDER",
    ("Treatment", "Drug"): "TREATMENT_DRUG",
    ("Treatment", "Dosage"): "TREATMENT_DOSAGE",
    ("Treatment", "Route"): "TREATMENT_ROUTE",
    ("Treatment", "Freq"): "TREATMENT_FREQ",
    ("Treatment", "Duration"): "TREATMENT_DURATION",
    ("Treatment", "Time_elapsed"): "TREATMENT_TIME_ELAPSED",
    ("Treatment", "Disorder"): "TREATMENT_DISORDER",
}

# Label names as used by the authors (question templates "Treatment.Drug in adverse event").
AUTHORS_LABEL_NAME = {
    "SUBJECT": "Subject",
    "TREATMENT": "Treatment",
    "EFFECT": "Effect",
    "SUBJECT_AGE": "Subject.Age",
    "SUBJECT_GENDER": "Subject.Gender",
    "SUBJECT_RACE": "Subject.Race",
    "SUBJECT_POPULATION": "Subject.Population",
    "SUBJECT_DISORDER": "Subject.Disorder",
    "TREATMENT_DRUG": "Treatment.Drug",
    "TREATMENT_DOSAGE": "Treatment.Dosage",
    "TREATMENT_ROUTE": "Treatment.Route",
    "TREATMENT_FREQ": "Treatment.Freq",
    "TREATMENT_DURATION": "Treatment.Duration",
    "TREATMENT_TIME_ELAPSED": "Treatment.Time_elapsed",
    "TREATMENT_DISORDER": "Treatment.Disorder",
}

# Short human readable names, used in prompts and definitions.
READABLE_NAME = {
    "SUBJECT": "subject",
    "TREATMENT": "treatment",
    "EFFECT": "effect",
    "SUBJECT_AGE": "age",
    "SUBJECT_GENDER": "gender",
    "SUBJECT_RACE": "race",
    "SUBJECT_POPULATION": "population",
    "SUBJECT_DISORDER": "disorder of the subject",
    "TREATMENT_DRUG": "drug",
    "TREATMENT_DOSAGE": "dosage",
    "TREATMENT_ROUTE": "route",
    "TREATMENT_FREQ": "frequency",
    "TREATMENT_DURATION": "duration",
    "TREATMENT_TIME_ELAPSED": "time elapsed",
    "TREATMENT_DISORDER": "disorder targeted by the treatment",
}

# Label definitions (paraphrased from the PHEE annotation scheme, Sun et al. 2022).
# Used by the "definitions" question-template variant and by the LLM prompts.
DEFINITIONS = {
    "ADE": (
        "An adverse drug event: a harmful or unintended reaction that occurred "
        "after a drug or therapy was used."
    ),
    "PTE": (
        "A potential therapeutic event: a beneficial or intended effect of a drug "
        "or therapy, such as an improvement of a disorder."
    ),
    TRIGGER: (
        "The trigger is the word or short phrase that most clearly signals that the "
        "event happened, e.g. 'induced', 'developed', 'associated', 'after', 'treated'."
    ),
    "SUBJECT": (
        "The subject is the patient or group of patients who experienced the event, "
        "including the words that describe them (age, gender, race, number, disorder)."
    ),
    "TREATMENT": (
        "The treatment is the drug therapy involved in the event, including the drug "
        "and its details such as dosage, route, frequency, duration and the disorder "
        "it was given for."
    ),
    "EFFECT": (
        "The effect is the outcome of the treatment: the adverse reaction for an adverse "
        "event, or the therapeutic outcome for a potential therapeutic event."
    ),
    "SUBJECT_AGE": "The age of the subject, e.g. '52-year-old', 'elderly', 'children'.",
    "SUBJECT_GENDER": "The gender of the subject, e.g. 'woman', 'man', 'male'.",
    "SUBJECT_RACE": "The race or ethnicity of the subject, e.g. 'Black', 'Caucasian'.",
    "SUBJECT_POPULATION": "The number of subjects, e.g. 'two', '15', 'a series of'.",
    "SUBJECT_DISORDER": "A disorder or condition that the subject has.",
    "TREATMENT_DRUG": "The name of a drug used in the treatment.",
    "TREATMENT_DOSAGE": "The dose of the drug, e.g. '50 mg', 'high-dose'.",
    "TREATMENT_ROUTE": "How the drug was given, e.g. 'intravenous', 'oral', 'topical'.",
    "TREATMENT_FREQ": "How often the drug was given, e.g. 'daily', 'twice a week'.",
    "TREATMENT_DURATION": "How long the treatment lasted, e.g. 'for 3 weeks'.",
    "TREATMENT_TIME_ELAPSED": (
        "The time between starting the treatment and the event, e.g. '4 weeks after'."
    ),
    "TREATMENT_DISORDER": "The disorder that the treatment was given for.",
}

# Priorities used when the gold annotation of one sentence has overlapping spans
# that cannot be placed in a single tag sequence (lower number = placed first).
MAIN_TAG_PRIORITY = {
    TRIGGER: 0,
    "TREATMENT": 1,
    "EFFECT": 2,
    "SUBJECT": 3,
}

# Sub-arguments: the more frequent label wins (counts in the revised data, all splits).
# Matters mostly for identical spans annotated with two labels in the revised data:
# SUBJECT_DISORDER = TREATMENT_DISORDER (1185 spans), DURATION = TIME_ELAPSED (206 spans).
SUB_TAG_PRIORITY = {
    label: idx
    for idx, label in enumerate([
        "TREATMENT_DRUG",          # 6053
        "TREATMENT_DISORDER",      # 1708
        "SUBJECT_DISORDER",        # 1564
        "SUBJECT_AGE",             # 705
        "TREATMENT_ROUTE",         # 632
        "SUBJECT_GENDER",          # 570
        "TREATMENT_DOSAGE",        # 468
        "SUBJECT_POPULATION",      # 458
        "TREATMENT_TIME_ELAPSED",  # 349
        "TREATMENT_DURATION",      # 267
        "TREATMENT_FREQ",          # 114
        "SUBJECT_RACE",            # 65
    ])
}

LEVEL_OF = {TRIGGER: "trigger"}
LEVEL_OF.update({label: "main" for label in MAIN_LABELS})
LEVEL_OF.update({label: "sub" for label in SUB_LABELS})

ALL_ARGUMENT_LABELS = MAIN_LABELS + SUB_LABELS
