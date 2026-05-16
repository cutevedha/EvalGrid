# Data Augmentation - Generates variations of test cases for robustness testing
# Creates paraphrases, typos, noise, and other variations to test AI system robustness

from typing import List, Dict, Any
import random
import re


# ============================================================================
# PARAPHRASING
# ============================================================================

def paraphrase_text(text: str, strategy: str = "synonym") -> str:
    """
    Create a paraphrased version of text

    Strategies:
    - synonym: Replace words with synonyms
    - reorder: Shuffle sentence order
    - expand: Add filler text

    Args:
        text: Text to paraphrase
        strategy: Paraphrasing strategy to use

    Returns:
        Paraphrased version of text
    """
    if strategy == "synonym":
        # Replace words with synonyms
        synonyms = {
            "good": ["excellent", "great", "fine", "wonderful"],
            "bad": ["poor", "terrible", "awful", "horrible"],
            "help": ["assist", "aid", "support", "facilitate"],
            "understand": ["comprehend", "grasp", "recognize", "perceive"],
            "important": ["significant", "crucial", "vital", "essential"],
        }

        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?;:')
            if word_lower in synonyms:
                words[i] = random.choice(synonyms[word_lower])

        return ' '.join(words)

    elif strategy == "reorder":
        # Shuffle sentence order
        sentences = text.split('. ')
        random.shuffle(sentences)
        return '. '.join(sentences)

    elif strategy == "expand":
        # Add filler text
        return text + " " + _generate_filler()

    return text


def _generate_filler() -> str:
    """Generate filler text to expand output"""
    fillers = [
        "This is important.",
        "Please note.",
        "Additionally,",
        "Furthermore,",
        "In conclusion,",
    ]
    return random.choice(fillers)


# ============================================================================
# TYPO INJECTION
# ============================================================================

def inject_typos(text: str, error_rate: float = 0.05) -> str:
    """
    Introduce typos and character errors into text

    Tests if AI systems are robust to misspellings

    Args:
        text: Text to corrupt
        error_rate: Fraction of words to introduce errors in (0.05 = 5%)

    Returns:
        Text with introduced typos
    """
    words = text.split()
    num_errors = max(1, int(len(words) * error_rate))

    for _ in range(num_errors):
        if not words:
            break
        idx = random.randint(0, len(words) - 1)
        word = words[idx]

        if len(word) > 2:
            # Choose random error type
            error_type = random.choice(['swap', 'delete', 'insert', 'replace'])

            if error_type == 'swap':
                # Swap adjacent characters
                char_idx = random.randint(0, len(word) - 2)
                word = word[:char_idx] + word[char_idx+1] + word[char_idx] + word[char_idx+2:]

            elif error_type == 'delete':
                # Delete a character
                char_idx = random.randint(0, len(word) - 1)
                word = word[:char_idx] + word[char_idx+1:]

            elif error_type == 'insert':
                # Insert a random character
                char_idx = random.randint(0, len(word))
                word = word[:char_idx] + random.choice('abcdefghijklmnopqrstuvwxyz') + word[char_idx:]

            elif error_type == 'replace':
                # Replace a character
                char_idx = random.randint(0, len(word) - 1)
                word = word[:char_idx] + random.choice('abcdefghijklmnopqrstuvwxyz') + word[char_idx+1:]

            words[idx] = word

    return ' '.join(words)


# ============================================================================
# CASE VARIATION
# ============================================================================

def change_case(text: str, case_type: str = "lower") -> str:
    """
    Change the case of text

    Args:
        text: Text to modify
        case_type: Case type (lower, upper, title, random)

    Returns:
        Text with modified case
    """
    if case_type == "lower":
        return text.lower()
    elif case_type == "upper":
        return text.upper()
    elif case_type == "title":
        return text.title()
    elif case_type == "random":
        # Randomly change case of each character
        return ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in text)
    return text


# ============================================================================
# NOISE INJECTION
# ============================================================================

def add_noise(text: str, noise_type: str = "whitespace") -> str:
    """
    Add various types of noise to text

    Args:
        text: Text to corrupt
        noise_type: Type of noise (whitespace, punctuation, special_chars)

    Returns:
        Text with added noise
    """
    if noise_type == "whitespace":
        # Add extra whitespace
        return re.sub(r'\s+', '  ', text)

    elif noise_type == "punctuation":
        # Add random punctuation
        punctuation = '.,!?;:'
        words = text.split()
        for i in range(len(words)):
            if random.random() > 0.7:
                words[i] += random.choice(punctuation)
        return ' '.join(words)

    elif noise_type == "special_chars":
        # Insert special characters
        chars_to_add = ['@', '#', '$', '%', '^', '&', '*']
        text_list = list(text)
        for i in range(len(text_list)):
            if random.random() > 0.9:
                text_list[i] = random.choice(chars_to_add)
        return ''.join(text_list)

    return text


# ============================================================================
# LANGUAGE TRANSLATION (SIMPLE)
# ============================================================================

def translate_to_language(text: str, language: str = "spanish") -> str:
    """
    Translate text to another language (simple word replacement)

    Tests multilingual robustness

    Args:
        text: Text to translate
        language: Target language (spanish, french, german)

    Returns:
        Translated text (simple word-level translation)
    """
    simple_translations = {
        "spanish": {
            "hello": "hola",
            "world": "mundo",
            "thank": "gracias",
            "please": "por favor",
            "help": "ayuda",
        },
        "french": {
            "hello": "bonjour",
            "world": "monde",
            "thank": "merci",
            "please": "s'il vous plaît",
            "help": "aide",
        },
        "german": {
            "hello": "hallo",
            "world": "welt",
            "thank": "danke",
            "please": "bitte",
            "help": "hilfe",
        },
    }

    if language not in simple_translations:
        return text

    translations = simple_translations[language]
    words = text.lower().split()
    translated = [translations.get(word, word) for word in words]
    return ' '.join(translated)


# ============================================================================
# EDGE CASE GENERATION
# ============================================================================

def generate_edge_cases(base_text: str) -> List[str]:
    """
    Generate edge cases of text

    Creates variations that test boundary conditions

    Args:
        base_text: Base text to generate edge cases from

    Returns:
        List of edge case variations
    """
    edge_cases = [
        base_text,  # Original
        base_text.strip(),  # Stripped
        base_text.upper(),  # Uppercase
        base_text.lower(),  # Lowercase
        base_text + " ",  # Trailing space
        " " + base_text,  # Leading space
        base_text.replace(" ", ""),  # No spaces
        base_text * 2,  # Repeated
        "",  # Empty
        " ",  # Single space
    ]
    return edge_cases


# ============================================================================
# DATASET AUGMENTATION
# ============================================================================

def augment_dataset(test_cases: List[Dict[str, Any]], augmentation_factor: int = 2) -> List[Dict[str, Any]]:
    """
    Augment a dataset by creating variations of test cases

    Args:
        test_cases: Original test cases
        augmentation_factor: How many times to multiply the dataset

    Returns:
        Augmented dataset with variations
    """
    augmented = test_cases.copy()

    for test_case in test_cases:
        for _ in range(augmentation_factor - 1):
            augmented_case = test_case.copy()
            augmented_case["id"] = f"{test_case['id']}_aug_{len(augmented)}"

            # Choose random augmentation strategy
            strategy = random.choice(["paraphrase", "typos", "case", "noise"])

            if strategy == "paraphrase" and "input" in augmented_case:
                augmented_case["input"] = paraphrase_text(augmented_case["input"])

            elif strategy == "typos" and "input" in augmented_case:
                augmented_case["input"] = inject_typos(augmented_case["input"], 0.05)

            elif strategy == "case" and "input" in augmented_case:
                augmented_case["input"] = change_case(augmented_case["input"], "random")

            elif strategy == "noise" and "input" in augmented_case:
                augmented_case["input"] = add_noise(augmented_case["input"])

            augmented.append(augmented_case)

    return augmented


# ============================================================================
# ADVERSARIAL VARIANT GENERATION
# ============================================================================

def create_adversarial_variants(text: str, num_variants: int = 5) -> List[str]:
    """
    Create adversarial variants of text

    Combines multiple augmentation techniques

    Args:
        text: Base text
        num_variants: Number of variants to generate

    Returns:
        List of adversarial variants
    """
    variants = [text]

    for _ in range(num_variants - 1):
        variant = text
        # Randomly apply augmentations
        if random.random() > 0.5:
            variant = paraphrase_text(variant)
        if random.random() > 0.5:
            variant = inject_typos(variant, 0.03)
        if random.random() > 0.5:
            variant = change_case(variant, random.choice(["lower", "upper", "title"]))
        variants.append(variant)

    return variants
