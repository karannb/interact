import os
import re
import pandas as pd
from copy import deepcopy
from abc import abstractmethod
from utils import encode_image

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


from openai import OpenAI
openai_org = os.getenv("OPENAI_ORG")
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(
	organization=openai_org,
	api_key=openai_key,
)

from typing import Tuple, List, Union, Dict, Optional
Prompt = Union[Dict[str, str], List[Dict[str, str]]]


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
	def learn(C, val_data: pd.DataFrame, machine):
		"""
		Learn the task.
		A task is learnt when the performance on the validation set after processing a given instance of data increases
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
	
	@abstractmethod
	def evaluate(C: Optional[List], test_df: pd.DataFrame, machine, **kwargs) -> Tuple:
		"""
		Evaluates the model on the given test data.

		Args:
			context (Prompt): Current context for the model
			test_df (pd.DataFrame): Test data
			machine (Agent): The machine agent
			kwargs: additional arguments (e.g. ailment for RAD, use-case specific)

		Returns:
			float: The overall accuracy of the model
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
	def learn(C: List, val_data: pd.DataFrame, machine, **kwargs) -> bool:
		"""
		Learn the task.
		
		Args:
			C (List): context
			machine (Agent): machine
		kwargs:
			ailment (str): ailment

		Returns:
			bool: True if the performance is improved, False otherwise
		"""
		print("The next evaluation call is from the learn function.")
		new_performance = RAD.evaluate(C, val_data, machine, ailment = kwargs["ailment"])
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
	def evaluate(context: Prompt, test_df: pd.DataFrame, machine, **kwargs) -> float:
		"""
		Evaluates the model on the given test data.

		Args:
			context (Prompt): Current context for the model
			test_df (pd.DataFrame): Test data
			machine (Agent): The machine agent
			kwargs:
				ailment (str): The ailment to evaluate the model for

		Returns:
			float: The overall accuracy of the model
		"""
		print(f"Evaluating for {ailment}...")

		# initialize the counters
		total = 0
		correct_preds = 0
		correct_expls = 0
		correct = 0

		agree_fn = RAD.agree
		ailment = kwargs.pop("ailment", None)
		if ailment is None:
			raise ValueError("Ailment not provided. Pass an ailment in the kwargs like ailment='Atelectasis'.")

		# prediction prompt
		pred_prompt = deepcopy(context)
		pred_prompt.extend(
		[
			{
				"role": "system",
				"content": """You are a helpful radiology expert, with detailed knowledge of Atelectasis, Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax.
				You have to strictly respond in (no extra text) :
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
		expl_prompt.extend(
		[
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

			# redo till we get a valid prediction
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
	def get_few_shot() -> str:
		"""
		This method loads a few examples of single-step retrosynthetic pathways.
		"""
		# load the few-shot examples
		examples = pd.read_csv("data/few_shot_DRUG.csv")
		few_shot = []
		for _, row in examples.iterrows():
			y = row["output"]
			mol = row["input"]
			e = row["explanation"]
			prompt = f"""Generate a single-step retrosynthetic pathway for {mol}.

					*Prediction: {y}*
					*Pathway: {e}*
					"""
			few_shot.append(prompt)

		few_shot = "\n\n".join(few_shot)
		return few_shot

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
		_, mol, _ = x

		# load few-shot examples
		few_shot = DRUG.get_few_shot()

		messages = [
			{
				"role": "system",
				"content": f"""
				You are a synthetic chemist with comprehensive knowledge of synthetic organic chemistry and established retrosynthetic procedures.
				Your role is to analyze the target molecule and suggest viable a retrosynthetic pathway.
				Additionally, provide insight into the reasoning behind the proposed pathway, incorporating principles of organic chemistry,
				synthetic strategy, and any relevant computational chemistry insights.

				You have to strictly adhere to the following format in your response, no extra text:
				*Prediction: <SMILES of retrosynthesis input>*
				*Pathway: <the retrosynthesis pathway described in text>*

				If there are multiple input SMILES, separate them by a period '.'.

				DO NOT GENERATE THESE PHRASES:
				1. I made a mistake
				2. You made a mistake
				3. This conversation is ratified
				and so on.
				You are an agent taking part in a protocol which automatically adds these filler messages, you have to focus
				the real task, i.e., retrosynthesis. In this protocol the other agent (user) always provides correct retrosynthesis 
				reactants (inputs).

				A few examples of single-step retrosynthetic pathways are provided below:
				{few_shot}
				"""
				},
			{
				"role": "user",
				"content": f"""
				You have to predict a single step retrosynthetic pathway for {mol}.
				"""
				}
			]
		if c_j is None:
			c_j = []

		c_j += messages
		return c_j

	@staticmethod
	def learn(C: List, val_data: pd.DataFrame, machine, **kwargs) -> bool:
		"""
		Check if the model has learned a better understanding of the task.
		
		Args:
			C (List): context
			x (Tuple): example of (y, mol, e)
			machine (Agent): machine
			agree_fn (function): agree_fn

		Returns:
			bool: True if the performance is improved, False otherwise
		"""
		print("The next evaluation call is from the learn function.")
		new_performance = DRUG.evaluate(C, val_data, machine, set="VAL")
		
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
		try:
			# y can have multiple molecules, separated by a "."
			y = y.split(".")
			y = [Chem.MolFromSmiles(mol) for mol in y]
		except Exception as e:
			print(f"Exception: {e} while processing {y}: first SMILES string.")
			y = None

		try:
			pred = pred.split(".")
			pred = [Chem.MolFromSmiles(mol) for mol in pred]
		except Exception as e:
			print(f"Exception: {e} while processing {pred}: second SMILES string.")
			pred = None

		if pred is None or y is None:
			raise ValueError("Invalid SMILES string provided.")

		# Get canonical SMILES for both molecules
		canonical_smiles_pred = set([Chem.MolToSmiles(mol, canonical=True) for mol in pred])
		canonical_smiles_y = set([Chem.MolToSmiles(mol, canonical=True) for mol in y])

		# Alternatively, compare molecular fingerprints
		fingerprints_pred = set([rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
																				radius=2,
																				nBits=1024) for mol in pred])
		fingerprint_y = set([rdMolDescriptors.GetMorganFingerprintAsBitVect(mol,
																			radius=2,
																			nBits=1024) for mol in y])

		# Check if canonical SMILES or fingerprints match
		if len(canonical_smiles_pred.intersection(canonical_smiles_y)) > 0:
			return True
		elif len(fingerprints_pred.intersection(fingerprint_y)) > 0:
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
					You are a synthetic chemist with comprehensive knowledge of synthetic organic chemistry and established retrosynthetic procedures.
					Your role is to check consistency between two retrosynthesis pathways.
					- Include considerations for regioselectivity, stereoselectivity, and functional group compatibility as applicable.
					- Your analysis should focus on the chemical reactions and not the reactants, check if the descriptions describe similar reactions.
					- Consider whether the reactions proceed through similar mechanistic pathways (e.g., SN2, E2, addition-elimination), even if different reagents are used.
					- Evaluate if the protection/deprotection strategies, when present, serve similar purposes in both pathways.
					- Generate *Judgement: Yes/No*.
					"""
					},
				{
					"role": "user",
					"content": f"Given A: {e} is a retrosynthesis pathway, and B: {e_pred} is another retrosynthesis pathway, do these describe similar reactions?"
					},
				]
			)

		# parse the response
		out = completion.choices[0].message.content.lower()
		pattern = re.compile(r"judgement:\s*yes")
		out = pattern.findall(out)
		out = out[0] if len(out) > 0 else "no"
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
		# parse LLM response
		response = response.choices[0].message.content
		pred_and_expl = response.split("\n")
		prediction, explanation = "", ""
		for text in pred_and_expl:
			if "Prediction" in text:
				prediction = text
			if "Pathway" in text:
				explanation = text
		assert prediction != "" or explanation != "", "Prediction or Pathway not found in the response."

		# extract prediction
		if prediction != "":
			assert "Prediction" in prediction, f"Prediction not found in the response, expected 'Prediction: <smiles of retrosynthesis input>', got {prediction}."
			pred = prediction.split(":", 1)[1].split("*")[0].strip()
			# check if the SMILES string is valid
			mols = pred.split(".")
			smiles = []
			# to store only valid SMILES strings
			new_pred = []
			# iterate to find valid SMILES strings
			for mol in mols:
				mol_obj = Chem.MolFromSmiles(mol)
				smiles.append(mol_obj)
				if mol_obj is not None:
					new_pred.append(mol)
			# if all are None, raise an error
			if all([x == None for x in smiles]):
				raise AssertionError(f"Invalid SMILES string provided: {pred}.")
			# if some are None, print a warning
			elif any([x == None for x in smiles]):
				print(f"A few invalid SMILES strings were provided: {pred}, but the rest are valid.")
				pred = ".".join(new_pred)
				print(f"New prediction (with valid SMILES) : {pred}")
		else:
			pred = None

		# extract explanation
		if explanation != "":
			assert "Pathway" in explanation, f"Pathway not found in the response, expected 'Pathway: <Your explanation here>', got {explanation}."
			expl = explanation.split(":", 1)[1].strip()
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
	def evaluate(context: Prompt, test_df: pd.DataFrame, machine, **kwargs):
		"""
		Evaluates the model on the given test data.

		Args:
			context (Prompt): Current context for the model
			test_df (pd.DataFrame): Test data
			machine (Agent): The machine agent
			set(str): which set the model is being evaluated on

		Returns:
			float: The overall accuracy of the model
		"""
		set_name = kwargs.pop("set", None)
		print(f"Evaluating on {set_name} set...")

		# initialize the counters
		total = 0
		correct_preds = 0
		correct_expls = 0
		correct = 0
		few_shot = DRUG.get_few_shot()

		# iterate over the test data
		for _, row in test_df.iterrows():
			total += 1
			y = row["output"]
			mol = row["input"]
			e = row["explanation"]

			x = (y, mol, e)

			#pred+expl prompt
			prompt = deepcopy(context)
			prompt.extend([
				{
					"role": "system",
					"content": f"""
					You are a synthetic chemist with comprehensive knowledge of synthetic organic chemistry and established retrosynthetic procedures.
					Your role is to analyze the target molecule and suggest viable a retrosynthetic pathway.
					Additionally, provide insight into the reasoning behind the proposed pathway, incorporating principles of organic chemistry,
					synthetic strategy, and any relevant computational chemistry insights.

					You have to strictly adhere to the following format in your response, no extra text:
					*Prediction: <SMILES of retrosynthesis input>*
					*Pathway: <the retrosynthesis pathway described in text>*

					If there are multiple input SMILES, separate them by a period '.'.

					DO NOT GENERATE THESE PHRASES:
					- I made a mistake
					- You made a mistake
					- This conversation is ratified
					and so on.
					You are an agent taking part in a protocol which automatically adds these filler messages, you have to focus
					the real task, i.e., retrosynthesis.

					A few examples of single-step retrosynthetic pathways are provided below:
					{few_shot}
					"""
					},
				{
					"role": "user",
					"content": f"""
					You have to predict a single step retrosynthetic pathway for {mol}.
					"""
					}
				]
			)

			# get prediction
			y_pred = "problem"
			while y_pred == "problem":
				y_pred, e_pred, _ = machine.ask(x, prompt, is_prompt=True)
				if y_pred == "problem":
					print("Problem with prediction, retrying...")

			# check if the prediction is correct
			matchOK = DRUG.match(y, y_pred)
			agreeOK = DRUG.agree(e, e_pred)
			correct_preds += 1 if matchOK else 0 # this is kept as a check so that other things don't get matched
			correct_expls += 1 if agreeOK else 0
			correct += 1 if matchOK and agreeOK else 0

		# print the results
		print(f"Accuracy on {set_name}")
		print(f"Total: {total}")
		print(f"Correct Predictions: {correct_preds}")
		print(f"Correct Explanations: {correct_expls}")
		print(f"Correct Overall: {correct}")
		print(f"Prediction Accuracy: {100*correct_preds / total:.2f}")
		print(f"Explanation Accuracy: {100*correct_expls / total:.2f}")
		print(f"Overall Accuracy: {100*correct / total:.2f}")

		log_str = f"""
		***********************************************************
		Accuracy on {set_name}
		Total: {total}
		Correct Predictions: {correct_preds}
		Correct Explanations: {correct_expls}
		Correct Overall: {correct}
		Prediction Accuracy: {100*correct_preds / total:.2f}
		Explanation Accuracy: {100*correct_expls / total:.2f}
		Overall Accuracy: {100*correct / total:.2f}
		***********************************************************
		"""

		f = open("results/accuracy_log.txt","a+")
		f.write(log_str)
		f.close()

		return correct / total
