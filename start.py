from anthropic import Anthropic
from anthropic.types import Usage
from langfuse import Langfuse
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import instructor
import uuid

langfuse = Langfuse()
client = instructor.from_anthropic(Anthropic())


class ClassificationOutput(BaseModel):
    classification: str
    explanation: str


categories = ["BEST", "MID", "WORST"]
customer_information = (
    "Brandon is an avid cat lover and has a thing for parrots. He doesn't prefer dogs"
)
business_niche = "Ecommerce store that sells dog toys"
prompt = f"Classify the customer: {customer_information}\nHere are the categories:{str(categories)}\nHere is the business niche: {business_niche}"


trace_id = str(uuid.uuid4)
trace = langfuse.trace(name="customer_classification")
generation = trace.generation(
    trace_id=trace_id, name="classification", model="claude-3-haiku-20240307"
)

output, raw = client.completions.create_with_completion(
    model="claude-3-haiku-20240307",
    messages=[{"role": "user", "content": prompt}],
    response_model=ClassificationOutput,
    max_tokens=4000,
)
usage: Usage = raw.usage
generation.end(
    output=output,
    input={"role": "user", "content": prompt},
    usage={"input": usage.input_tokens, "output": usage.output_tokens},
)
trace.update(input={"role": "user", "content": prompt}, output=output)
