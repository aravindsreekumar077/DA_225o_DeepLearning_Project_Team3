import random
import json
import csv
from typing import List, Dict

NUM_SAMPLES = 1000  # Number of samples to generate

basic_templates = {
    "add": [
        "What is {a} plus {b}?",
        "Can you add {a} and {b} together?",
        "Calculate the sum of {a} and {b}.",
        "Add {a} and {b}.",
        "What's the result of adding {a} with {b}?",
        "Find the total of {a} and {b}.",
        "How much is {a} plus {b}?",
        "Sum up {a} and {b} for me.",
        "Add the numbers {a} and {b}.",
        "What do you get if you add {a} and {b}?",
        "Could you please add {a} and {b} for me?"
    ],
    "subtract": [
        "What is {a} minus {b}?",
        "Subtract {b} from {a}.",
        "Can you calculate {a} minus {b}?",
        "What's the difference between {a} and {b}?",
        "Find the result of {a} minus {b}.",
        "How much is {a} less {b}?",
        "Calculate the subtraction of {a} and {b}.",
        "What’s {a} take away {b}?",
        "Take {b} away from {a}.",
        "Find the difference of {a} and {b}.",
        "I need to subtract {b} from {a}, can you help?"
    ],
    "multiply": [
        "Multiply {a} by {b}.",
        "What is the product of {a} and {b}?",
        "Calculate {a} times {b}.",
        "Multiply {a} with {b}.",
        "What do you get if you multiply {a} and {b}?",
        "Find the multiplication result of {a} and {b}.",
        "How much is {a} multiplied by {b}?",
        "Compute the product of {a} and {b}.",
        "Multiply these numbers: {a} and {b}.",
        "Calculate the multiplication of {a} and {b}.",
        "Let’s multiply {a} with {b} and see the result."
    ],
    "divide": [
        "What is {a} divided by {b}?",
        "Divide {a} by {b}.",
        "Calculate the quotient of {a} and {b}.",
        "What do you get if you divide {a} by {b}?",
        "Find the result of {a} over {b}.",
        "How much is {a} divided by {b}?",
        "Compute the division of {a} by {b}.",
        "Divide these numbers: {a} and {b}.",
        "What's the ratio of {a} to {b}?",
        "Calculate {a} divided by {b}.",
        "How much do you get when dividing {a} by {b}?"
    ],
    "sqrt": [
        "What is the square root of {a}?",
        "Calculate the square root of {a}.",
        "Find sqrt({a})."
    ],
    "power": [
        "What is {a} raised to the power of {b}?",
        "Calculate {a} ** {b}.",
        "Raise {a} to the {b}th power."
    ]
}

matrix_templates = {
    "add": [
        "Add the matrices {m1} and {m2}.",
        "What is the sum of {m1} and {m2}?",
        "Calculate the matrix addition of {m1} and {m2}.",
        "Perform addition on matrices {m1} and {m2}.",
        "Find the result of adding {m1} to {m2}.",
        "Add matrices {m1} and {m2}.",
        "What's the result of {m1} plus {m2}?",
        "Could you add the two matrices {m1} and {m2} and show me the output?",
        "What is the sum of the matrices {m1} and {m2}?"
    ],
    "subtract": [
        "Subtract {m2} from {m1}.",
        "What is the difference between {m1} and {m2}?",
        "Calculate the matrix subtraction of {m1} and {m2}.",
        "Perform subtraction on matrices {m1} and {m2}.",
        "What do you get if you subtract {m2} from {m1}?",
        "Subtract matrix {m2} from matrix {m1}.",
        "Find the difference between these matrices: {m1} and {m2}."
    ],
    "multiply": [
        "Multiply the matrices {m1} and {m2}.",
        "What is the product of {m1} and {m2}?",
        "Calculate the matrix multiplication of {m1} and {m2}.",
        "Perform multiplication on matrices {m1} and {m2}.",
        "Find the result of multiplying {m1} with {m2}.",
        "Compute the product of matrices {m1} and {m2}.",
        "Multiply these matrices: {m1} and {m2}.",
        "Calculate the matrix product of {m1} and {m2}.",
        "What do you get if you multiply {m1} by {m2}?",
        "Multiply matrix {m1} with matrix {m2}.",
        "Can you multiply matrix {m1} with matrix {m2}?"
    ]
}

def generate_basic_sample(op: str) -> Dict[str, str]:
    a = random.randint(1, 100)
    b = random.randint(1, 100)
    prompt_template = random.choice(basic_templates[op])

    if op == "sqrt":
        prompt = prompt_template.format(a=a)
        xml = f"""
<calc>
  <expression>
    <type>unary</type>
    <operation>{op}</operation>
    <operands>
      <operand>{a}</operand>
    </operands>
  </expression>
</calc>""".strip()
    else:
        prompt = prompt_template.format(a=a, b=b)
        xml = f"""
<calc>
  <expression>
    <type>binary</type>
    <operation>{op}</operation>
    <operands>
      <operand>{a}</operand>
      <operand>{b}</operand>
    </operands>
  </expression>
</calc>""".strip()
    return {"prompt": prompt, "output": xml}

def generate_matrix(rows=2, cols=2) -> List[List[int]]:
    return [[random.randint(1, 9) for _ in range(cols)] for _ in range(rows)]

def matrix_to_xml(matrix: List[List[int]]) -> str:
    return "\n".join(f"<row>{' '.join(map(str, row))}</row>" for row in matrix)

def generate_matrix_sample(op: str) -> Dict[str, str]:
    m1 = generate_matrix()
    m2 = generate_matrix()
    m1_str = f"{m1}"
    m2_str = f"{m2}"
    prompt_template = random.choice(matrix_templates[op])
    prompt = prompt_template.format(m1=m1_str, m2=m2_str)

    xml = f"""
<calc>
  <expression>
    <type>matrix</type>
    <operation>{op}</operation>
    <operands>
      <matrix>
        {matrix_to_xml(m1)}
      </matrix>
      <matrix>
        {matrix_to_xml(m2)}
      </matrix>
    </operands>
  </expression>
</calc>""".strip()

    return {"prompt": prompt, "output": xml}

def generate_dataset(n_samples=NUM_SAMPLES) -> List[Dict[str, str]]:
    dataset = []

    ops_basic = list(basic_templates.keys())
    ops_matrix = list(matrix_templates.keys())

    for _ in range(n_samples):
        if random.random() < 0.85:  # 85% basic, 15% matrix
            op = random.choice(ops_basic)
            dataset.append(generate_basic_sample(op))
        else:
            op = random.choice(ops_matrix)
            dataset.append(generate_matrix_sample(op))

    return dataset

def save_to_csv(data: List[Dict[str, str]], filename: str):
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["prompt", "output"])
        writer.writeheader()
        for row in data:
            writer.writerow(row)

if __name__ == "__main__":
    data = generate_dataset()

    # Save as JSONL
    with open("calc_dataset.jsonl", "w", encoding="utf-8") as f:
        for item in data:
            json.dump(item, f)
            f.write("\n")

    # Save as CSV
    save_to_csv(data, "calc_dataset.csv")

    print(f"Dataset saved as calc_dataset.jsonl and calc_dataset.csv with {NUM_SAMPLES} samples.")
