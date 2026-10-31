"""Prompts, adapted from the reference notebooks."""


def product_description_messages(item: str) -> list[dict]:
    system = (
        "You are a product marketer for a company that makes nutrition supplements.\n"
        "Balance your product descriptions to attract customers, optimize SEO, and\n"
        "stay within accurate advertising guidelines.\n"
        "Product descriptions have to be 3-5 sentences.\n"
        "Provide only the product description with no preamble."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Write a product description for a {item}."}]


def nursery_rhyme_messages(animal: str) -> list[dict]:
    system = (
        "You are writing nursery rhymes about animals for a children's book.\n"
        "Each poem should be 3-5 lines long.\n"
        "Return only the poem, without any introduction or preamble."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"Write a poem about a {animal}."}]


def acrostic_messages(animal: str, target: str, next_letter: str = "") -> list[dict]:
    system = (
        "You are a poet writing acrostic poems about animals for a children's book.\n"
        "The first letters of the lines of an acrostic poem spell out a word.\n"
        "Write one short line per letter. Respond with only the poem, no title or preamble.\n\n"
        "For example, an acrostic poem about rabbits that spells QUICK:\n\n"
        "Quietly, the rabbit bides its time\n"
        "Under the garden deck\n"
        "In wait for\n"
        "Carrot greens,\n"
        "Kale, and parsley."
    )
    user = (f"Write an acrostic poem about a {animal}. "
            f"The first letters of its lines must spell {target.upper()}, one line per letter.")
    if next_letter:
        user += f" The next line must begin with the letter {next_letter.upper()}."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
