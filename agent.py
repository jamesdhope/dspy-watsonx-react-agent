import dspy
from dsp.utils import deduplicate
from dspy.datasets import HotPotQA
from dspy.predict.retry import Retry
from dspy.teleprompt import BootstrapFewShot
from dspy.evaluate.evaluate import Evaluate
from dspy.primitives.assertions import assert_transform_module, backtrack_handler
from MilvusRM import MilvusRM
from dspy.datasets import HotPotQA
from dsp import LM
import dspy

from watsonxModel import watsonx

dspy.settings.configure(lm=watsonx, trace=[], temperature=0.7)
retriever_model = MilvusRM(collection_name="wikipedia_articles",uri="http://localhost:19530")
dspy.settings.configure(rm=retriever_model)

# signature for our RAG Agent
class GenerateAnswer(dspy.Signature):
    """You are a helpful, friendly assistant that can answer questions"""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(prefix="Reasoning: Let's think step by step.",desc="often between 10 and 20 words")

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()

        self.retrieve = retriever_model #dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ReAct(signature=GenerateAnswer) #dspy.ReAct(GenerateAnswer) #dspy.Predict(GenerateAnswer) 
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer) 

# Uncompiled module prediction
answer = dspy.Predict(GenerateAnswer)(context="", question="Who is Alan Turing?")
print(answer)

# Load the dataset
dataset = HotPotQA(train_seed=1, train_size=20, eval_seed=2023, dev_size=50, test_size=0)
# Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
trainset = [x.with_inputs('question') for x in dataset.train]
devset = [x.with_inputs('question') for x in dataset.dev]
len(trainset), len(devset)

#def metric(example: dspy.Example, prediction, trace=None):
#        
#    transcript, answer, summary = example.transcript, example.summary, prediction.summary
#    
#    with dspy.context(lm=watsonx):
#        # This next line is the one that results in the error when called from the optimizer.
#        content_eval = dspy.Predict(Assess)(summary=summary, assessment_question=\
#                            f"Is the assessed text a good summary of this transcript, capturing all the important details?\n\n{transcript}?")
#    return content_eval.to_lower().endswith('yes')

# Define the signature for automatic assessments.
#class Assess(dspy.Signature):
#    """Assess the quality of a tweet along the specified dimension."""

#    assessed_text = dspy.InputField()
#    assessment_question = dspy.InputField()
#    assessment_answer = dspy.OutputField(desc="Yes or No")

#def metric(example, pred, trace=None):

#    engaging = "Does the assessed text make for a self-contained, engaging tweet?"
#    correct = f"The text should answer `{example.question}` with `{pred}`. Does the assessed text contain this answer?"
    
#    print("correct",correct)

#    with dspy.context(lm=watsonx):
#        correct =  dspy.Predict(Assess)(assessed_text=pred, assessment_question=correct)
#        engaging = dspy.Predict(Assess)(assessed_text=pred, assessment_question=engaging)

#    correct, engaging = [m.assessment_answer.lower() == 'yes' for m in [correct, engaging]]
#    score = (correct + engaging) if correct and (len(example.question) <= 280) else 0

#    if trace is not None: return score >= 2
#    return score / 2.0

# Validation logic: check that the predicted answer is correct.
# Also check that the retrieved context does actually contain that answer.
def validate_context_and_answer(example, pred, trace=None):
    answer_EM = dspy.evaluate.answer_exact_match(example, pred)
    answer_PM = dspy.evaluate.answer_passage_match(example, pred)
    return answer_EM and answer_PM

# Set up a basic teleprompter, which will compile our RAG program.
teleprompter = BootstrapFewShot(metric=validate_context_and_answer)

# Compile the RAG program
compiled_rag = teleprompter.compile(student=RAG(), trainset=trainset)

# Compiled module prediction
answer = dspy.Predict(GenerateAnswer)(context="",question="Who is Alan Turing?")
print(answer)

#dspy.candidate_programs

#compiled_rag.inspect_history(n=3)