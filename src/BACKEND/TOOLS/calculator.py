import re
from sympy import sympify, SympifyError

def word_to_number(text):
    text = text.strip()
    if text.isdigit():
        return int(text)

    ones = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
    }
    teens = {
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16,
        "seventeen": 17, "eighteen": 18, "nineteen": 19
    }
    tens = {
        "twenty": 20, "thirty": 30, "forty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90
    }

    parts = text.lower().split()
    number = 0
    for i, word in enumerate(parts):
        if word in teens:
            number += teens[word]
        elif word in tens:
            number += tens[word]
            if i + 1 < len(parts) and parts[i + 1] in ones:
                number += ones[parts[i + 1]]
                break
        elif word in ones:
            number += ones[word]
    return number

def normalize_expression(sentence):
    sentence = sentence.lower().strip()

    sentence = re.sub(r"(?:the )?add (\w+(?: \w+)*) and (\w+(?: \w+)*)", r"\1 + \2", sentence)
    sentence = re.sub(r"(?:the )?sum of (\w+(?: \w+)*) and (\w+(?: \w+)*)", r"\1 + \2", sentence)
    sentence = re.sub(r"(?:the )?product of (\w+(?: \w+)*) and (\w+(?: \w+)*)", r"\1 * \2", sentence)
    sentence = re.sub(r"(?:the )?difference of (\w+(?: \w+)*) and (\w+(?: \w+)*)", r"\1 - \2", sentence)
    sentence = re.sub(r"(?:the )?quotient of (\w+(?: \w+)*) and (\w+(?: \w+)*)", r"\1 / \2", sentence)

    sentence = re.sub(r"(?:the )?sum between (\w+(?: \w+)*) and (\w+(?: \w+)*)", r"\1 + \2", sentence)
    sentence = re.sub(r"(?:the )?product between (\w+(?: \w+)*) and (\w+(?: \w+)*)", r"\1 * \2", sentence)
    sentence = re.sub(r"(?:the )?difference between (\w+(?: \w+)*) and (\w+(?: \w+)*)", r"\1 - \2", sentence)
    sentence = re.sub(r"(?:the )?quotient between (\w+(?: \w+)*) and (\w+(?: \w+)*)", r"\1 / \2", sentence)

    sentence = re.sub(r"(plus|add|added to)", "+", sentence)
    sentence = re.sub(r"(minus|subtract|less)", "-", sentence)
    sentence = re.sub(r"(times|multiply|multiplied by)", "*", sentence)
    sentence = re.sub(r"(divided by|over|by)", "/", sentence)

    sentence = re.sub(r"\b(what is|calculate|find|the|between|please|display|show|write|give|result of)\b", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence)

    return sentence.strip()

def convert_words_to_expression(sentence):
    normalized = normalize_expression(sentence)
    for op in ['+', '-', '*', '/']:
        if op in normalized:
            parts = normalized.split(op)
            if len(parts) == 2:
                break
    else:
        return None, "❌ Operator not found"

    try:
        num1 = word_to_number(parts[0].strip())
        num2 = word_to_number(parts[1].strip())
        expression = f"{num1} {op} {num2}"
        return expression, None
    except Exception as e:
        return None, f"❌ Could not convert: {str(e)}"

def evaluate_expression(user_input):
    lowered = user_input.lower().strip()

    if lowered.startswith("calculate "):
        expr = user_input.split("calculate", 1)[-1].strip()
        try:
            result = sympify(expr).evalf()
            return f"🧮 Result of `{expr}` is `{result}`"
        except (SympifyError, Exception) as e:
            return f"❌ Error evaluating expression:\n`{str(e)}`"

    if any(op in lowered for op in [
        "plus", "minus", "divided", "times", "multiply", "subtract",
        "difference", "between", "quotient", "sum", "product", "add"
    ]):
        expr, err = convert_words_to_expression(user_input)
        if err:
            return err
        try:
            result = sympify(expr).evalf()
            return f"🧮 Result of `{expr}` is `{result}`"
        except Exception as e:
            return f"❌ Evaluation error: {str(e)}"

    return None  # Not a math expression
