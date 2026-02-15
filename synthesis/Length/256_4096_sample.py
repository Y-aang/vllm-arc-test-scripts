import json
import random
from typing import List

# Alphabet: b~z (25 letters total)
LETTERS = [chr(i) for i in range(ord('b'), ord('z') + 1)]

def generate_block(block_id: int) -> str:
    """
    Generate a 16-character block.
    Creates unique blocks through different permutations.
    Generated in order: first block is all b's, second block is 15 b + 1 c, etc.
    """
    # Convert block_id to a 16-digit base-25 number
    # This allows generating 25^16 different block combinations
    block = []
    remaining = block_id
    for i in range(16):
        letter_idx = remaining % 25
        block.append(LETTERS[letter_idx])
        remaining = remaining // 25
    return ''.join(block)

def generate_sentence(target_length: int, sentence_id: int) -> str:
    """
    Generate a sentence of specified length.
    target_length: target length (allows ± 16 tolerance)
    sentence_id: sentence ID, used to generate different block combinations
    """
    # Use sentence_id as random seed to ensure each sentence ID produces a unique sentence
    rng = random.Random(sentence_id)
    
    # Calculate how many blocks are needed
    num_blocks = target_length // 16
    # Generate blocks
    blocks = []
    for i in range(num_blocks):
        # Use sentence_id and block position to generate unique blocks
        # Use a larger multiplier to ensure different sentences use different block ranges
        block_id = sentence_id * 10000 + i
        block = generate_block(block_id)
        blocks.append(block)
    
    # Combine all blocks
    sentence = ''.join(blocks)
    
    # Add random error (+-16), using rng to ensure reproducibility
    length_error = rng.randint(-16, 16)
    actual_length = target_length + length_error
    actual_length = max(1, actual_length)  # Ensure at least 1 character
    
    # If actual length differs from target length, adjust sentence length
    if len(sentence) < actual_length:
        # If too short, add random characters
        extra = actual_length - len(sentence)
        sentence += ''.join(rng.choices(LETTERS, k=extra))
    elif len(sentence) > actual_length:
        # If too long, truncate
        sentence = sentence[:actual_length]
    
    # Convert sentence to space-separated format (add space after each character)
    return ' '.join(sentence)

def generate_all_sentences(num_256: int = 500, num_4096: int = 5):
    """
    Generate all required sentences.
    """
    # Generate num_256 sentences of length 256
    sentences_256 = []
    for i in range(num_256):
        sentence = generate_sentence(256, i)
        sentences_256.append(sentence)
    
    # Generate num_4096 sentences of length 4096
    sentences_4096 = []
    for i in range(num_4096):
        sentence = generate_sentence(4096, i)
        sentences_4096.append(sentence)
    
    return sentences_256, sentences_4096

def sample_requests(
    sentences_256: List[str],
    sentences_4096: List[str],
    num_requests: int = 3072,
    prob_256: float = 0.862,
):
    """
    Sample requests.
    86.2% probability of selecting a length-256 sentence, 13.8% probability of selecting a length-4096 sentence.
    """
    requests = []
    for i in range(num_requests):
        # Select length based on probability
        if random.random() < prob_256:
            # Select a length-256 sentence
            sentence = random.choice(sentences_256)
        else:
            # Select a length-4096 sentence
            print(f"Selected a length-4096 sentence", i)
            sentence = random.choice(sentences_4096)
        requests.append(sentence)
    return requests

def main():
    # Set random seed to ensure reproducibility
    random.seed(42)
    
    # Current requirement: 2000 length-256 sentences, 5 length-4096 sentences
    num_256 = 2000
    num_4096 = 5
    prob_256 = num_256 / (num_4096 + num_256)
    num_requests = 4096

    print("Generating sentences...")
    sentences_256, sentences_4096 = generate_all_sentences(
        num_256=num_256, num_4096=num_4096
    )
    print(f"Generated {len(sentences_256)} length-256 sentences")
    print(f"Generated {len(sentences_4096)} length-4096 sentences")
    
    # Verify sentence lengths
    print("\nVerifying sentence lengths...")
    lengths_256 = [len(s.replace(' ', '')) for s in sentences_256]
    lengths_4096 = [len(s.replace(' ', '')) for s in sentences_4096]
    print(f"Length-256 sentence range: {min(lengths_256)} - {max(lengths_256)}")
    print(f"Length-4096 sentence range: {min(lengths_4096)} - {max(lengths_4096)}")
    
    print("\nSampling requests...")
    requests = sample_requests(
        sentences_256,
        sentences_4096,
        num_requests=num_requests,
        prob_256=prob_256,
    )
    print(f"Sampled {len(requests)} requests")
    
    # Summarize sampling results
    count_256 = sum(1 for r in requests if len(r.replace(' ', '')) < 3000)
    count_4096 = len(requests) - count_256
    print(f"Length-256 sentence count: {count_256} ({count_256/len(requests)*100:.1f}%)")
    print(f"Length-4096 sentence count: {count_4096} ({count_4096/len(requests)*100:.1f}%)")
    
    print("\nWriting to file...")
    with open('sentences.json', 'w', encoding='utf-8') as f:
        json.dump(requests, f, indent=4, ensure_ascii=False)
    
    print("Done!")

if __name__ == '__main__':
    main()

