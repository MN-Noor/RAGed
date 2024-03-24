import dspy
from dotenv import load_dotenv
import os
load_dotenv()
apikey=os.getenv("OPENAI_API")
class Context:
    def __init__(self, doc):
        self.doc = doc

class GenerateAnswer(dspy.Signature):
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="quesion part should be of one or two lines but answer part of the query should be long and descriptive")

class QnaGenerator:
    def __init__(self, model="gpt-3.5-turbo", api_key=None):
        self.llm = dspy.OpenAI(model=model, api_key=api_key)
        dspy.settings.configure(lm=self.llm)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def generate(self, context, question):
        prediction = self.generate_answer(context=context, question=question)
        return prediction.answer

# Sample usage:
doc = """
"signaling that investments in the supply chain sector remain robust, Pando, a startup developing fulfillment management technologies, today announced that it raised $30 million in a Series B round, bringing its total raised to $45 million.Iron Pillar and Uncorrelated Ventures led the round, with participation from existing investors Nexus Venture Partners, Chiratae Ventures and Next47. CEO and founder Nitin Jayakrishnan says that the new capital will be put toward expanding Pando’s global sales, marketing and delivery capabilities.“We will not expand into new industries or adjacent product areas,” he told TechCrunch in an email interview. “Great talent is the foundation of the business — we will continue to augment our teams at all levels of the organization. Pando is also open to exploring strategic partnerships and acquisitions with this round of funding.”Pando was co-launched by Jayakrishnan and Abhijeet Manohar, who previously worked together at iDelivery, an India-based freight tech marketplace — and their first startup. The two saw firsthand manufacturers, distributors and retailers were struggling with legacy tech and point solutions to understand, optimize and manage their global logistics operations — or at least, that’s the story Jayakrishnan tells.“Supply chain leaders were trying to build their own tech and throwing people at the problem,” he said. “This caught our attention — we spent months talking to and building for enterprise users at warehouses, factories, freight yards and ports and eventually, in 2018, decided to start Pando to solve for global logistics through a software-as-a-service platform offering.”'"
"""
context = Context(doc)
pando_generator = QnaGenerator(api_key=apikey)

context = context.doc
question = "Generate question and its respective answer from given context"
result = pando_generator.generate(context, question)
print(result)
