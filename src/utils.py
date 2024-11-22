import os
import base64
import pandas as pd
from copy import deepcopy
from openai import OpenAI


openai_org = os.getenv("OPENAI_ORG")
openai_key = os.getenv("OPENAI_KEY")
client = OpenAI(
    organization=openai_org,
    api_key=openai_key,
)

from typing import List, Union, Dict, Callable
Prompt = Union[Dict[str, str], List[Dict[str, str]]]


def encode_image(image_path):
    """
    Default encoding for images is base64.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def summarize(report: str, ailment: str) -> str:
    """
    Generates a summary of the report, in context of the ailment.

    Args:
        report (str): A radiology report
        ailment (str): The ailment to summarize the report for

    Returns:
        str: The summary of the report
    """
    # NOTE: for this task, 3.5 is just as good as any advanced model
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "system", 
                "content": "You are a radiology expert, with detailed knowledge of Atelectasis, Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax."
                },
            {
                "role": "user",
                "content": "Summarize the given radiology report in context of " 
                + ailment + 
                ". Also, you must (requirement) omit information about the patient age, name, as well as any links. You can also skip the 'report by' information, basically anything not related to the ailment."
                + " Only include information that explicitly mentions the ailment or is close to such a mention."
                + " Strictly do not write 'summary' anywhere, i.e., summarize the report as if you are generating it."
                + " The report is as follows: "
                },
            {
                "role": "user",
                "content": report
                }
            ]
        )

    # parse the response
    summary = completion.choices[0].message.content

    return summary


def evaluate(context: Prompt, test_df: pd.DataFrame, ailment: str, agree_fn: Callable) -> None:
    """
    Evaluates the model on the given test data.

    Args:
        context (Prompt): Current context for the model
        test_df (pd.DataFrame): Test data
        ailment (str): The ailment to evaluate the model for
        agree_fn (Callable): Function to check if the explanation agrees with the prediction
    """

    # initialize the counters
    total = 0
    correct_preds = 0
    correct_expls = 0
    correct = 0

    # prediction prompt
    pred_prompt = deepcopy(context)
    pred_prompt.append(
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
    )

    expl_prompt = deepcopy(context)
    expl_prompt.append(
        {
            "role": "system",
            "content": """You are a helpful radiology expert, with detailed knowledge of Atelectasis, Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax.
            You have to strictly response in, no extra text:
            *Explanation: <Your explanation here>*
            """
        },
        {
            "role": "user",
            "content": f"Given the following chest XRay, you have to explain the presence of {ailment}."
        }
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
        prediction = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=pred_prompt + [img_msg]
        )
        prediction = prediction.choices[0].message.content

        # if label != ailment, we don't care about explanation, only prediction
        if label != ailment:
            matchOK = prediction.lower() == "no"
            correct_preds += 1 if matchOK else 0
            correct_expls += 1 if matchOK else 0
            correct += 1 if matchOK else 0
        else:
            if prediction.lower() == "no":
                # the ailment IS present, but the model predicted it as absent
                pass
            else:
                # get explanation
                explanation = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=expl_prompt + [img_msg]
                )
                explanation = explanation.choices[0].message.content

                # check if the prediction is correct
                matchOK = prediction.lower() == "yes"
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

    return


def evaluate_many(context: Prompt, test_data: Dict[str, pd.DataFrame], agree_fn: Callable) -> None:
    """
    Evaluates the model on multiple test data.

    Args:
        context (Prompt): Current context for the model
        test_data (Dict[str, pd.DataFrame]): Test data for multiple ailments
        agree_fn (Callable): Function to check if the explanation agrees with the prediction
    """

    for ailment, test_df in test_data.items():
        evaluate(context, test_df, ailment, agree_fn)

    return
