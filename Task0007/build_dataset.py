import json


evaluation_data = [

    {
        "question": "Who is the main character of The Alchemist?",
        "ground_truth": "The main character of The Alchemist is Santiago, a young shepherd."
    },

    {
        "question": "What is Santiago's profession at the beginning of the story?",
        "ground_truth": "Santiago is a shepherd boy at the beginning of the story."
    },

    {
        "question": "Where does Santiago dream of finding treasure?",
        "ground_truth": "Santiago dreams of finding treasure near the Egyptian pyramids."
    },

    {
        "question": "Who tells Santiago about his Personal Legend?",
        "ground_truth": "Melchizedek, the King of Salem, tells Santiago about his Personal Legend."
    },

    {
        "question": "What are Urim and Thummim?",
        "ground_truth": "Urim and Thummim are two stones given to Santiago to help him read omens."
    },

    {
        "question": "How does Santiago help improve the crystal merchant's business?",
        "ground_truth": "Santiago suggests building a crystal display to attract more customers and increase sales."
    },

    {
        "question": "Who is Fatima?",
        "ground_truth": "Fatima is a woman from the oasis whom Santiago falls in love with."
    },

    {
        "question": "What is the Soul of the World?",
        "ground_truth": "The Soul of the World is the spiritual force that connects all living things and the universe."
    },

    {
        "question": "Why is Fatima willing to let Santiago continue his journey?",
        "ground_truth": "Fatima believes Santiago should follow his Personal Legend and trusts he will return."
    },

    {
        "question": "Where was the treasure actually hidden?",
        "ground_truth": "The treasure was hidden near the abandoned church in Spain where Santiago first dreamed about it."
    },

    {
        "question": "Who is the Englishman?",
        "ground_truth": "The Englishman is a traveler studying alchemy and searching for the alchemist."
    },

    {
        "question": "What happens to Santiago when he arrives in Tangier?",
        "ground_truth": "Santiago is robbed of all his money shortly after arriving in Tangier."
    },

    {
        "question": "Why does the crystal merchant refuse to go to Mecca?",
        "ground_truth": "The crystal merchant fears that fulfilling his dream of visiting Mecca would leave him with nothing else to hope for."
    },

    {
        "question": "What is a Personal Legend?",
        "ground_truth": "A Personal Legend is a person's true purpose or destiny in life."
    },

    {
        "question": "What is the central message of The Alchemist?",
        "ground_truth": "The novel teaches people to follow their dreams, pursue their Personal Legend, and trust the journey."
    }

]


with open(
    "evaluation/evaluation_dataset.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        evaluation_data,
        f,
        indent=4,
        ensure_ascii=False
    )


print("Evaluation dataset created successfully!")
print(f"Total questions: {len(evaluation_data)}")