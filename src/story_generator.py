"""
Synthetic story generator for character tracking experiments.

Generates narratives with a controllable number of characters, each with
trackable attributes (location, possession) that change through the story.
Ground truth is programmatically verified.
"""

import random
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

# Distinctive character names (diverse, unambiguous, easy to distinguish)
CHARACTER_NAMES = [
    "Alice", "Borislav", "Chen", "Diana", "Emeka", "Fatima", "Gustavo", "Hana",
    "Ivan", "Jia", "Kwame", "Layla", "Marco", "Nadia", "Oscar", "Priya",
    "Quinn", "Rashid", "Suki", "Tomas", "Uma", "Viktor", "Wanda", "Xavier",
    "Yuki", "Zara", "Amara", "Bjorn", "Camille", "Dmitri", "Elena", "Felix",
    "Greta", "Hugo", "Ines", "Jasper", "Kira", "Luca", "Mina", "Nikolai",
    "Olga", "Pavel", "Rosa", "Sergei", "Thea", "Uri", "Vera", "Werner",
    "Xena", "Yusuf", "Zelda", "Adira", "Bruno", "Chiara", "Dante", "Ewa",
    "Faris", "Giselle", "Henrik", "Isla", "Jovan", "Katya", "Leif", "Marta",
    "Nils", "Opal", "Petra", "Riku", "Signe", "Tariq", "Ulrike", "Viggo",
]

LOCATIONS = [
    "the kitchen", "the garden", "the library", "the attic", "the basement",
    "the living room", "the bedroom", "the workshop", "the courtyard", "the balcony",
    "the hallway", "the study", "the dining room", "the terrace", "the cellar",
    "the tower", "the chapel", "the stables", "the market", "the bakery",
    "the harbor", "the plaza", "the bridge", "the park", "the inn",
    "the castle gate", "the throne room", "the dungeon", "the observatory", "the greenhouse",
    "the forge", "the tavern",
]

POSSESSIONS = [
    "a golden key", "a leather journal", "a silver compass", "a red scarf",
    "a wooden flute", "a brass lantern", "a velvet pouch", "a crystal vial",
    "a copper ring", "an iron dagger", "a silk ribbon", "a jade pendant",
    "a bronze mirror", "an ivory comb", "a pewter goblet", "a woven basket",
    "a feathered quill", "a polished stone", "a carved box", "a glass orb",
    "a tattered map", "a sealed letter", "a porcelain figurine", "a pearl necklace",
    "an emerald brooch", "a steel chain", "a bone whistle", "a painted fan",
    "a sapphire ring", "a cloth bundle", "a pocket watch", "a music box",
]

# Templates for state-change sentences
MOVE_TEMPLATES = [
    "{name} walked to {location}.",
    "{name} headed over to {location}.",
    "{name} made their way to {location}.",
    "{name} went to {location}.",
    "{name} strolled into {location}.",
]

PICKUP_TEMPLATES = [
    "{name} picked up {item}.",
    "{name} found {item} and took it.",
    "{name} grabbed {item}.",
    "{name} collected {item}.",
]

GIVE_TEMPLATES = [
    "{name} gave {item} to {recipient}.",
    "{name} handed {item} over to {recipient}.",
    "{name} passed {item} to {recipient}.",
]

DROP_TEMPLATES = [
    "{name} set down {item}.",
    "{name} left {item} behind.",
    "{name} put {item} aside.",
]


@dataclass
class CharacterState:
    name: str
    location: str
    possession: Optional[str]
    introduction_order: int


@dataclass
class StoryInstance:
    num_characters: int
    story_text: str
    characters: list  # list of dicts with final state
    initial_states: list  # list of dicts with initial state
    events: list  # list of event descriptions
    questions: list  # list of question dicts
    ground_truth: list  # list of answer dicts


def generate_story(num_characters: int, seed: int, state_changes_per_char: int = 2) -> StoryInstance:
    """Generate a synthetic story with num_characters characters."""
    rng = random.Random(seed)

    # Select characters, locations, and possessions
    names = rng.sample(CHARACTER_NAMES[:len(CHARACTER_NAMES)], num_characters)
    # Need enough unique locations: initial + changes
    total_locs_needed = num_characters * (1 + state_changes_per_char)
    available_locs = list(LOCATIONS)
    rng.shuffle(available_locs)
    # We'll reuse locations if needed but try to keep them distinct
    all_locs = available_locs * ((total_locs_needed // len(available_locs)) + 1)

    available_items = list(POSSESSIONS)
    rng.shuffle(available_items)
    all_items = available_items * ((num_characters * 2 // len(available_items)) + 1)

    # Initialize characters
    characters = []
    initial_states = []
    loc_idx = 0
    item_idx = 0
    for i, name in enumerate(names):
        loc = all_locs[loc_idx]
        loc_idx += 1
        item = all_items[item_idx]
        item_idx += 1
        cs = CharacterState(name=name, location=loc, possession=item, introduction_order=i)
        characters.append(cs)
        initial_states.append({
            "name": name,
            "location": loc,
            "possession": item,
            "introduction_order": i,
        })

    # Build introduction paragraph
    intro_lines = []
    for cs in characters:
        intro_lines.append(
            f"{cs.name} was in {cs.location}, carrying {cs.possession}."
        )
    intro_paragraph = " ".join(intro_lines)

    # Generate state-change events
    events = []
    event_sentences = []

    for _ in range(state_changes_per_char):
        # Shuffle character order for each round of changes
        char_order = list(range(num_characters))
        rng.shuffle(char_order)

        for ci in char_order:
            cs = characters[ci]
            # Decide event type: move location or change possession
            event_type = rng.choice(["move", "move", "pickup"])  # bias toward moves

            if event_type == "move":
                # Pick a new location different from current
                new_loc = rng.choice([l for l in LOCATIONS if l != cs.location])
                template = rng.choice(MOVE_TEMPLATES)
                sentence = template.format(name=cs.name, location=new_loc)
                events.append({"character": cs.name, "type": "move", "old": cs.location, "new": new_loc})
                cs.location = new_loc
                event_sentences.append(sentence)

            elif event_type == "pickup":
                # Drop current item and pick up a new one
                old_item = cs.possession
                new_item = all_items[item_idx % len(all_items)]
                item_idx += 1
                # Make sure it's different
                while new_item == old_item:
                    new_item = all_items[item_idx % len(all_items)]
                    item_idx += 1

                drop_sentence = rng.choice(DROP_TEMPLATES).format(name=cs.name, item=old_item)
                pickup_sentence = rng.choice(PICKUP_TEMPLATES).format(name=cs.name, item=new_item)
                events.append({"character": cs.name, "type": "swap_item", "old": old_item, "new": new_item})
                cs.possession = new_item
                event_sentences.append(drop_sentence + " " + pickup_sentence)

    # Build full story text
    # Add some paragraph breaks for readability
    mid = len(event_sentences) // 2
    events_text_1 = " ".join(event_sentences[:mid])
    events_text_2 = " ".join(event_sentences[mid:])

    story_text = f"{intro_paragraph}\n\n{events_text_1}\n\n{events_text_2}"

    # Generate questions and ground truth
    questions = []
    ground_truth = []

    for cs in characters:
        # Location question
        questions.append({
            "character": cs.name,
            "attribute": "location",
            "question": f"Where is {cs.name} now?",
        })
        ground_truth.append({
            "character": cs.name,
            "attribute": "location",
            "answer": cs.location,
        })

        # Possession question
        questions.append({
            "character": cs.name,
            "attribute": "possession",
            "question": f"What is {cs.name} carrying?",
        })
        ground_truth.append({
            "character": cs.name,
            "attribute": "possession",
            "answer": cs.possession,
        })

    # Build final state dicts
    final_states = []
    for cs in characters:
        final_states.append({
            "name": cs.name,
            "location": cs.location,
            "possession": cs.possession,
            "introduction_order": cs.introduction_order,
        })

    return StoryInstance(
        num_characters=num_characters,
        story_text=story_text,
        characters=final_states,
        initial_states=initial_states,
        events=events,
        questions=questions,
        ground_truth=ground_truth,
    )


def generate_dataset(
    character_counts: list[int] = [2, 4, 8, 16, 32],
    stories_per_count: int = 5,
    state_changes_per_char: int = 2,
    base_seed: int = 42,
) -> list[StoryInstance]:
    """Generate a full dataset of stories across character counts."""
    dataset = []
    for nc in character_counts:
        for trial in range(stories_per_count):
            seed = base_seed * 1000 + nc * 100 + trial
            story = generate_story(nc, seed=seed, state_changes_per_char=state_changes_per_char)
            dataset.append(story)
    return dataset


if __name__ == "__main__":
    # Quick test
    dataset = generate_dataset(character_counts=[2, 4], stories_per_count=2)
    for story in dataset:
        print(f"=== {story.num_characters} characters ===")
        print(story.story_text[:300])
        print(f"Questions: {len(story.questions)}")
        print(f"First Q: {story.questions[0]}")
        print(f"First A: {story.ground_truth[0]}")
        print()
