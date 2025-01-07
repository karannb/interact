import os
import pandas as pd
from openai import OpenAI
from copy import deepcopy
from abc import abstractmethod
from utils import encode_image, are_molecules_same
from typing import Tuple, List, Union, Dict, Optional, Callable

Prompt = Union[Dict[str, str], List[Dict[str, str]]]

openai_org = os.getenv("OPENAI_ORG")
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(
	organization=openai_org,
	api_key=openai_key,
)


class Task:
	"""
	Base class for a task (e.g. RAD, DRUG);
	the task decides 4 functions, namely, assemble_prompt, match, agree, and parse_response.
	"""
	def __init__(self):
		pass

	@abstractmethod
	def assemble_prompt(x, c_j) -> Prompt:
		"""
		Assemble prompt for the LLM.

		Args:
			x: example
			c_j: context

		Returns:
			Prompt: prompt
		"""
		raise NotImplementedError("Subclass must implement abstract method.")

	@abstractmethod
	def learn(C, ailment, machine, agree_fn):
		"""
		Learn the task.
		"""
		raise NotImplementedError("Subclass must implement abstract method.")

	@abstractmethod
	def match(y, pred) -> bool:
		"""
		Match the prediction with the example.

		Args:
			y: ground truth
			pred: prediction

		Returns:
			bool: True if the prediction matches the example, False otherwise
		"""
		raise NotImplementedError("Subclass must implement abstract method.")

	@abstractmethod
	def agree(e, e_pred) -> bool:
		"""
		Check if the explanation agrees with the prediction.

		Args:
			e: explanation
			e_pred: explanation

		Returns:
			bool: True if the explanation agrees with the prediction, False otherwise
		"""
		raise NotImplementedError("Subclass must implement abstract method.")

	@abstractmethod
	def parse_response(response, C: Optional[List]) -> Tuple:
		"""
		Parse the response from the LLM.

		Args:
			response (str): response from the LLM
			C (List): context

		Returns:
			Tuple: prediction and explanation
		"""
		raise NotImplementedError("Subclass must implement abstract method.")


class RAD(Task):
	"""
	Task for Radiology Diagnosis.
	"""
	def __init__(self):
		super().__init__()

	@staticmethod
	def assemble_prompt(x, c_j) -> Prompt:
		"""
		Assemble prompt for the LLM.

		Args:
			x: example
			c_j: context

		Returns:
			Prompt: prompt
		"""
		y, img, _ = x
		encoded_img = encode_image(img)
		messages = [
			{
				"role": "system",
				"content": """
				Pretend that you are a helpful radiology expert, with detailed knowledge of Atelectasis, Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax.
				It is also known that the user's predictions are always correct, i.e., ground truth.
				You have to strictly adhere to the following format in your response, no extra text:
				*Prediction: Yes/No*
				*Explanation: <Your explanation here>*
				"""
			},
			{
				"role": "user",
				"content": f"""
				Given the following chest XRay, you have to predict the presence of {y}.
				"""
			},
			{
				"role": "user",
				"content": [
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{encoded_img}"
						}
					}
				]
			}
		]
		if c_j is None:
			c_j = []

		c_j += messages
		return c_j

	@staticmethod
	def learn(C: List, val_data:pd.DataFrame, ailment: str, machine, agree_fn: Callable) -> bool:
		"""
		Learn the task.
		
		Args:
			C (List): context
			ailment (str): ailment
			machine (Agent): machine
			agree_fn (function): agree_fn

		Returns:

		"""

		new_performance = RAD.evaluate(C, val_data, ailment, machine, agree_fn)
		# return true if the performance is improved
		if machine.performance <= new_performance:
			machine.performance = new_performance
			return True
		else:
			return False

	@staticmethod
	def match(y, pred) -> bool:
		"""
		Match the prediction with the example.

		Args:
			y: ground truth
			pred: prediction

		Returns:
			bool: True if the prediction matches the example, False otherwise
		"""
		pred = str(pred).lower()
		y = str(y).lower()
		if "yes" in pred or y == pred:
			return True
		else:
			return False

	@staticmethod
	def agree(e, e_pred) -> bool:
		"""
		Check if the explanation agrees with the prediction.

		Args:
			e: explanation
			e_pred: explanation

		Returns:
			bool: True if the explanation agrees with the prediction, False otherwise
		"""
		# NOTE: for this task, we need to use GPT-4, 3.5 is not enough
		completion = client.chat.completions.create(
			model="gpt-4o",
			temperature=0.0,
			seed=42,
			messages=[
				{
					"role": "system",
					"content": """
					You are a radiology expert, with detailed knowledge of Atelectasis, Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax.
					Your task is to check consistency between two given diagnoses/explanations of an XRay.
					1. Ignore any personal patient information mentioned in either diagnosis/explanation, e.g. age, name, etc.
					2. Consider consistency in terms of the symptoms only and not the causes, e.g. if a report mentions xyz can be
					diagnosed from follow-up and another report just mentions xyz, then this is no problem, it's not necessary to mention follow-up.
					3. VERY IMPORTANT, your answer should be the same if the two reports are swapped, i.e., independent of the order of the two reports.
					4. Respond only in Yes/No."""
				},
				{
					"role": "user",
					"content": f"Given A: {e} is a diagnosis/explanation of an XRay, and B: {e_pred} is another diagnosis/explanation of an XRay, are these two consistent?"
				},
			]
		)

		# parse the response
		out = completion.choices[0].message.content.lower()
		if "yes" in out:
			return True
		else:
			return False

	@staticmethod
	def parse_response(response, C: Optional[List]) -> Tuple:
		"""
		Parse the response from the LLM.

		Args:
			response (str): response from the LLM
			C (List): context

		Returns:
			Tuple: prediction and explanation
		"""
		response = response.choices[0].message.content
		pred_and_expl = response.split("\n")
		prediction, explanation = "", ""
		for text in pred_and_expl:
			if "Prediction" in text:
				prediction = text
			if "Explanation" in text:
				explanation = text
		assert prediction != "" or explanation != "", "Prediction or Explanation not found in the response."
		if prediction != "":
			assert "Prediction" in prediction, "Prediction not found in the response, expected 'Prediction: Yes/No', got " + prediction
			pred = prediction.split(":")[1].strip()
		else:
			pred = None
		if explanation != "":
			assert "Explanation" in explanation, "Explanation not found in the response, expected 'Explanation: <Your explanation here>', got " + explanation
			expl = explanation.split(":")[1].strip()
		else:
			expl = None

		# add to the context
		response_conv = {
			"role": "assistant",
			"content": response
		}
		if C != None:
			C.append(response_conv)
		else:
			C = [response_conv]

		return pred, expl, C

	@staticmethod
	def evaluate(context: Prompt, test_df: pd.DataFrame, ailment: str, machine, agree_fn: Callable) -> float:
		"""
		Evaluates the model on the given test data.

		Args:
			context (Prompt): Current context for the model
			test_df (pd.DataFrame): Test data
			ailment (str): The ailment to evaluate the model for
			machine (Agent): The machine agent
			agree_fn (Callable): Function to check if the explanation agrees with the prediction

		Returns:
			float: The overall accuracy of the model
		"""
		print(f"Evaluating for {ailment}...")

		# initialize the counters
		total = 0
		correct_preds = 0
		correct_expls = 0
		correct = 0

		# prediction prompt
		pred_prompt = deepcopy(context)
		pred_prompt.extend([
			{
				"role": "system",
				"content": """You are a helpful radiology expert, with detailed knowledge of Atelectasis, Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax.
				You have to strictly response in, no extra text:
				*Prediction: Yes/No*
				"""
			},
			{
				"role": "user",
				"content": f"Given the following chest XRay, you have to predict the presence of {ailment}."
			}
		]
		)

		expl_prompt = deepcopy(context)
		expl_prompt.extend([
			{
				# TODO: check prompt
				"role": "system",
				"content": """You are a helpful radiology expert, with detailed knowledge of Atelectasis, Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax.
				You have to strictly response in, no extra text:
				*Prediction: Yes*
				*Explanation: <Your explanation here>*
				"""
			},
			{
				"role": "user",
				"content": f"Given the following chest XRay, you have to explain the presence of {ailment}."
			},
			{
				"role": "assistant",
				"content": "*Prediction: Yes*"
			}
		]
		)

		# iterate over the test data
		for _, row in test_df.iterrows():
			total += 1
			report = row["report"]
			label = row["label"]
			img = encode_image(row["img_path"])

			# get prediction
			img_msg = {
				"role": "user",
				"content": [
					{
						"type": "image_url",
						"image_url": {
							"url": f"data:image/jpeg;base64,{img}"
						}
					}
				]
			}
			y = "problem"
			while y == "problem":
				y, _, _ = machine.ask(None, pred_prompt + [img_msg], is_prompt=True)
				if y == "problem":
					print("Problem with prediction, retrying...")

			prediction = y

			# if label != ailment, we don't care about explanation, only prediction
			if label != ailment:
				matchOK = "no" in prediction.lower()
				correct_preds += 1 if matchOK else 0
				correct_expls += 1 if matchOK else 0
				correct += 1 if matchOK else 0
			else:
				if "no" in prediction.lower():
					# the ailment IS present, but the model predicted it as absent
					pass
				else:
					# get explanation
					y = "problem"
					while y == "problem":
						y, explanation, _ = machine.ask(None, expl_prompt + [img_msg], is_prompt=True)
						if y == "problem":
							print("Problem with explanation, retrying...")

					# check if the prediction is correct
					matchOK = "yes" in prediction.lower()
					agreeOK = agree_fn(explanation, report)
					correct_preds += 1 if matchOK else 0 # this is kept as a check so that other things don't get matched
					correct_expls += 1 if agreeOK else 0
					correct += 1 if matchOK and agreeOK else 0

		# print the results
		print(f"Ailment: {ailment}")
		print(f"Total: {total}")
		print(f"Correct Predictions: {correct_preds}")
		print(f"Correct Explanations: {correct_expls}")
		print(f"Correct Overall: {correct}")
		print(f"Prediction Accuracy: {100*correct_preds / total:.2f}")
		print(f"Explanation Accuracy: {100*correct_expls / total:.2f}")
		print(f"Overall Accuracy: {100*correct / total:.2f}")

		return correct / total

class DRUG(Task):
	"""
	Task for Retrosynthesis.
	"""
	def __init__(self):
		super().__init__()

	@staticmethod
	def assemble_prompt(x, c_j) -> Prompt:
		"""
		Assemble prompt for the LLM.

		Args:
			x: example
			c_j: context

		Returns:
			Prompt: prompt
		"""
		y, _, _ = x
		messages = [
			{
				"role": "system",
				"content": """
				Pretend that you are a helpful retrosynthesis expert, with detailed knowledge of all known retrosynthesis procedures.
				It is also known that the user's predictions are always correct, i.e., ground truth.
				You have to strictly adhere to the following format in your response, no extra text:
				*Prediction: <smiles of retrosynthesis product>*
				*Explanation: <Your explanation here>*
				"""
			},
			{
				"role": "user",
				"content": f"""
				You have to predict the single step retrosynthesis of {y}.
				"""
			}
		]
		if c_j is None:
			c_j = []

		c_j += messages
		return c_j

	@staticmethod
	def learn(C: List, val_data:pd.DataFrame, machine, match_fn: Callable, agree_fn: Callable) -> bool:
		"""
		Learn the task.
		
		Args:
			C (List): context
			x (Tuple): example of (y, mol, e)
			machine (Agent): machine
			agree_fn (function): agree_fn

		Returns:

		"""

		new_performance = DRUG.evaluate(C,val_data,machine,match_fn,agree_fn)
		# return true if the performance is improved
		if machine.performance <= new_performance:
			machine.performance = new_performance
			return True
		else:
			return False

	@staticmethod
	def match(y, pred) -> bool:
		"""
		Match the prediction with the example.

		Args:
			y: ground truth
			pred: prediction

		Returns:
			bool: True if the prediction matches the example, False otherwise
		"""
		res = are_molecules_same(y, pred)
		if res:
			return True
		else:
			return False

	@staticmethod
	def agree(e, e_pred) -> bool:
		"""
		Check if the explanation agrees with the prediction.

		Args:
			e: explanation
			e_pred: explanation

		Returns:
			bool: True if the explanation agrees with the prediction, False otherwise
		"""
		# NOTE: for this task, we need to use GPT-4, 3.5 is not enough
		completion = client.chat.completions.create(
			model="gpt-4o",
			temperature=0.0,
			seed=42,
			messages=[
				{
					"role": "system",
					"content": """
					You are a retrosynthesis expert, with detailed knowledge of all known retrosynthesis procedures.
					Your task is to check consistency between two given explanations of a retrosynthesis procedure.
					1. VERY IMPORTANT, your answer should be the same if the two reports are swapped, i.e., independent of the order of the two reports.
					2. Respond only in Yes/No."""
				},
				{
					"role": "user",
					"content": f"Given A: {e} is an explanation of a retrosynthesis pathway, and B: {e_pred} is another explanation of a retrosynthesis pathway, are these two consistent?"
				},
			]
		)

		# parse the response
		out = completion.choices[0].message.content.lower()
		if "yes" in out:
			return True
		else:
			return False

	@staticmethod
	def parse_response(response, C: Optional[List]) -> Tuple:
		"""
		Parse the response from the LLM.

		Args:
			response (str): response from the LLM
			C (List): context

		Returns:
			Tuple: prediction and explanation
		"""
		response = response.choices[0].message.content
		pred_and_expl = response.split("\n")
		prediction, explanation = "", ""
		for text in pred_and_expl:
			if "Prediction" in text:
				prediction = text
			if "Explanation" in text:
				explanation = text
		assert prediction != "" or explanation != "", "Prediction or Explanation not found in the response."
		if prediction != "":
			assert "Prediction" in prediction, "Prediction not found in the response, expected 'Prediction: <smiles of retrosynthesis product>', got " + prediction
			pred = prediction.split(":")[1].strip()
		else:
			pred = None
		if explanation != "":
			assert "Explanation" in explanation, "Explanation not found in the response, expected 'Explanation: <Your explanation here>', got " + explanation
			expl = explanation.split(":")[1].strip()
		else:
			expl = None

		# add to the context
		response_conv = {
			"role": "assistant",
			"content": response
		}
		if C != None:
			C.append(response_conv)
		else:
			C = [response_conv]

		return pred, expl, C

	@staticmethod
	def evaluate(context: Prompt, test_df: pd.DataFrame, machine ,match_fn: Callable, agree_fn: Callable):
		"""
		Evaluates the model on the given test data.

		Args:
			context (Prompt): Current context for the model
			test_df (pd.DataFrame): Test data
			machine (Agent): The machine agent
			agree_fn (Callable): Function to check if the explanation agrees with the prediction

		Returns:
			float: The overall accuracy of the model
		"""

		# print(f"Evaluating for {ailment}...")

		# initialize the counters
		total = 0
		correct_preds = 0
		correct_expls = 0
		correct = 0

				
		# iterate over the test data
		for _, row in test_df.iterrows():
			total += 1
			y = row["input"]
			mol = row["output"]
			e = row["explanation"]

			x = (y,mol,e)

			#pred+expl prompt
			prompt = deepcopy(context)
			prompt.extend([
				{
					"role": "system",
					"content": """
					Pretend that you are a helpful retrosynthesis expert, with detailed knowledge of all known retrosynthesis procedures.
					It is also known that the user's predictions are always correct, i.e., ground truth.
					You have to strictly adhere to the following format in your response, no extra text:
					*Prediction: <smiles of retrosynthesis product>*
					*Explanation: <Your explanation here>*
					"""
				},
				{
					"role": "user",
					"content": f"""
					You have to predict the single step retrosynthesis of {y}.
					"""
				}
			]
			)

			# get prediction
			y_pred, e_pred, _ = machine.ask(x, prompt, is_prompt=True)
			
			
			# check if the prediction is correct
			matchOK = match_fn(mol,y_pred)
			agreeOK = agree_fn(e,e_pred)
			correct_preds += 1 if matchOK else 0 # this is kept as a check so that other things don't get matched
			correct_expls += 1 if agreeOK else 0
			correct += 1 if matchOK and agreeOK else 0



		# print the results
		print(f"Molecule: {y}")
		print(f"Total: {total}")
		print(f"Correct Predictions: {correct_preds}")
		print(f"Correct Explanations: {correct_expls}")
		print(f"Correct Overall: {correct}")
		print(f"Prediction Accuracy: {100*correct_preds / total:.2f}")
		print(f"Explanation Accuracy: {100*correct_expls / total:.2f}")
		print(f"Overall Accuracy: {100*correct / total:.2f}")

		return correct / total
