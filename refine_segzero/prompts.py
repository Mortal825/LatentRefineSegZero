# Shared prompts.
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant."
)


# Stage1 geometric query prompt. Keep this template unchanged because the
# trained geometric query model depends on this exact task style.
GEOMETRIC_QUERY_TEMPLATE = (
    "Please find '{Question}' with bbox and points."
    "Compare nearby objects carefully and pick the object that best matches the phrase."
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    "The final answer must be valid JSON with one bbox and two inner points: "
    "{{\"bbox\": [x1,y1,x2,y2], \"points_1\": [x,y], \"points_2\": [x,y]}}."
)


# Legacy reflection prompts used by geometric_query_model.py. They are kept for
# compatibility with the trained geometric/reflection execution path.
DECISION_REFLECTION_TEMPLATE = (
    "We already segmented one candidate for the referring expression.\n"
    "Question: {question}\n"
    "First answer: {first_answer}\n"
    "Mask summary: {mask_summary}\n"
    "Judge whether the first result should be accepted as the final geometric proposal.\n"
    "Return valid JSON inside <answer></answer> with the schema "
    "{{\"decision\": 1, \"reason\": \"...\"}} or {{\"decision\": 0, \"reason\": \"...\"}}.\n"
    "Use decision=1 for accept and decision=0 for reject."
)

REPAIR_REFLECTION_TEMPLATE = (
    "The first geometric proposal was rejected and needs one more correction.\n"
    "Question: {question}\n"
    "First answer: {first_answer}\n"
    "Decision reflection: {decision_answer}\n"
    "Mask summary: {mask_summary}\n"
    "Predict a better bbox, two inner points, and a confidence score for whether this repaired geometry "
    "is reliable enough to use with the aligned branch.\n"
    "Return valid JSON inside <answer></answer> with the schema "
    "{{\"bbox\": [x1,y1,x2,y2], \"points_1\": [x,y], \"points_2\": [x,y], \"confidence\": 0.0}}."
)


# Query-Reflect GRPO prompts.
QUERY_REFLECT_INIT_BOX_ANSWER_EXAMPLE = (
    "{\"bbox\": [10,100,200,210], \"points_1\": [30,110], \"points_2\": [35,180]}"
)

QUERY_REFLECT_REFLECT_ANSWER_EXAMPLE = (
    "{\"decision\":\"reject\",\"confidence\":0.23}"
)

QUERY_REFLECT_REFLECT_RL_ANSWER_EXAMPLE = (
    "{\"decision\":\"accept\"}"
)

QUERY_REFLECT_INIT_BOX_PROMPT_TEMPLATE = (
    "<image>\n"
    "Please find \"{Question}\" with bbox and points.\n"
    "Compare nearby objects carefully and pick the object that best matches the phrase.\n"
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.\n"
    "The final answer must be valid JSON with one bbox and two inner points:\n"
    "{Answer}"
)

QUERY_REFLECT_REFLECT_PROMPT_TEMPLATE = (
    "<image>\n<image>\n"
    "Question: \"{Question}\"\n"
    "Previous bbox: {PreviousBbox}\n"
    "The first image is the original image. The second image shows the previous bbox result.\n"
    "Judge whether the previous bbox correctly localizes the referred object.\n"
    "Set confidence to the estimated quality of the previous bbox: closer to 1.0 means better, closer to 0.0 means worse.\n"
    "Output <think>...</think> and <answer>...</answer>.\n"
    "The final answer must be valid JSON with exactly two keys: \"decision\" and \"confidence\".\n"
    "Use \"accept\" if the previous bbox is usable; use \"reject\" if it is not usable.\n"
    "Examples:\n"
    "<answer>{{\"decision\":\"accept\",\"confidence\":0.84}}</answer>\n"
    "<answer>{Answer}</answer>\n"
    "Do not output any other JSON keys."
)

QUERY_REFLECT_REFLECT_RL_PROMPT_TEMPLATE = (
   "<image>\n"
    "Question: \"{Question}\"\n"
    "You are an expert annotator of referred object localization. For an image and a referring question, you are given a red box and asked to evaluate whether the red box involves the referred target object.\n"
    "Reason 4 steps:\n"
    "1. Identify the target object described in the question.\n"
    "2. Zoom in on the red-box region and inspect it carefully. Identify all objects visible within the red box, including foreground, background, and partially visible objects. \n"
    "3. Check whether the important details in the question, such as relation to other objects, color, position, size and attribute, are consistent. We do not require the red box to be perfectly aligned with the object. However, the box should involve the referred target object.\n"
    "4. Give the final conclusion: \"accept\" or \"reject\". Do not include any explanation. Answer \"accept\" if no other visible object in the image that matches the description of the target object.  Answer \"reject\" only if you find a better candidate in your reasoning.\n"

    "Output exactly 4 reasoning steps in <think>...</think> and the final answer in <answer>...</answer>.\n"
    "The final answer must be valid JSON with exactly one key: \"decision\".\n"

    "Examples:\n"
    "Example 1:\n"
    "<think>1. The target object is the dog lying on the sofa. 2. The red box covers the dog, including its visible body region. 3. Some details are partially ambiguous, but no other visible object in the image matches the question better than the boxed dog. 4. Conclusion: accept</think><answer>{{\"decision\":\"accept\"}}</answer>\n"

    "Example 2:\n"
    "<think>1. The target object is the man on the left. 2. The red box covers a different person. 3. The position detail does not match because the boxed person is on the right side, while the question refers to the man on the left. 4. Conclusion: reject</think><answer>{{\"decision\":\"reject\"}}</answer>\n"

    "Now begin to decide whether the red box involves the referred target object in the question.\n"
)

# QUERY_REFLECT_REFLECT_RL_PROMPT_TEMPLATE = (
#     "<image>\n"
#     "Question: \"{Question}\"\n"
#     "The image contains a candidate region highlighted by a red box.\n"
#     "Your task is to decide whether the red box involves the referred target object in the question.\n"
#     "Reason 4 steps:\n"
#     "1. Identify the target object described in the question.\n"
#     "2. Zoom in on the red-box region and inspect it carefully. Identify all objects visible within the red box, including foreground, background, and partially visible objects. \n"
#     "3. Check whether the important details in the question, such as relation to other objects, color, position, size and attribute, are consistent.\n"
#     "4. Give the final conclusion: \"accept\" or \"reject\". Do not include any explanation. Answer \"accept\" if no other visible object in the image that matches the description of the target object.  Answer \"reject\" only if you find a better candidate in your reasoning.\n"

#     "Output exactly 4 reasoning steps in <think>...</think> and the final answer in <answer>...</answer>.\n"
#     "The final answer must be valid JSON with exactly one key: \"decision\".\n"

#     "Examples:\n"
#     "Example 1:\n"
#     "<think>1. The target object is the dog lying on the sofa. 2. The red box covers the dog, including its visible body region. 3. Some details are partially ambiguous, but no other visible object in the image matches the question better than the boxed dog. 4. Conclusion: accept</think><answer>{{\"decision\":\"accept\"}}</answer>\n"

#     "Example 2:\n"
#     "<think>1. The target object is the man on the left. 2. The red box covers a different person. 3. The position detail does not match because the boxed person is on the right side, while the question refers to the man on the left. 4. Conclusion: reject</think><answer>{{\"decision\":\"reject\"}}</answer>\n"

#     "Now begin to decide whether the red box involves the referred target object in the question.\n"
# )

#38.8
# QUERY_REFLECT_REFLECT_RL_PROMPT_TEMPLATE = (    
#     "<image>\n"
#     "Question: \"{Question}\"\n"
#     "The image contains a candidate region highlighted by a red box.\n"
#     "Your task is to decide whether the red box involves the referred target object in the question.\n"
#     "Reason 4 steps:\n"
#     "1. Identify the target object described in the question.\n"
#     "2. Zoom in on the red-box region and inspect it carefully. Identify all objects visible within the red box, including foreground, background, and partially visible objects. \n"
#     "3. Check whether the important details in the question, such as relation to other objects, color, position, size and attribute, are consistent.\n"
#     "4. Give the final conclusion: \"accept\" or \"reject\". Do not include any explanation. Answer \"accept\" if no other visible object in the image that matches the description of the target object.  Answer \"reject\" only if you find a better candidate in your reasoning.\n"

#     "Output exactly 4 reasoning steps in <think>...</think> and the final answer in <answer>...</answer>.\n"
#     "The final answer must be valid JSON with exactly one key: \"decision\".\n"

#     "Examples:\n"
#     "Example 1:\n"
#     "<think>1. The target object is the dog lying on the sofa. 2. The red box covers the dog, including its visible body region. 3. Some details are partially ambiguous, but no other visible object in the image matches the question better than the boxed dog. 4. Conclusion: accept</think><answer>{{\"decision\":\"accept\"}}</answer>\n"

#     "Example 2:\n"
#     "<think>1. The target object is the man on the left. 2. The red box covers a different person. 3. The position detail does not match because the boxed person is on the right side, while the question refers to the man on the left. 4. Conclusion: reject</think><answer>{{\"decision\":\"reject\"}}</answer>\n"

#     "Now begin to decide whether the red box involves the referred target object in the question.\n"
# )


## 45.4
# QUERY_REFLECT_REFLECT_RL_PROMPT_TEMPLATE = (    
#    "<image>\n"
#     "Question: \"{Question}\"\n"
#     "The image contains a candidate region highlighted by a red box.\n"
#     "Your task is to decide whether the red box involves the referred target object in the question.\n"
#     "Reason 4 steps:\n"
#     "1. Identify the target object described in the question.\n"
#     "2. Zoom in on the red-box region and inspect it carefully. Identify all objects visible within the red box, including foreground, background, and partially visible objects. \n"
#     "3. Check whether the important details in the question, such as relation to other objects and attribute, are consistent. We do not require the red box to be perfectly aligned with the object. However, the box should involve the referred target object.\n"
#     "4. Give the final conclusion: \"accept\" or \"reject\". Do not include any explanation. Answer \"accept\" if no other visible object in the image that matches the description of the target object.  Answer \"reject\" only if you find a better candidate in your reasoning.\n"

#     "Output exactly 4 reasoning steps in <think>...</think> and the final answer in <answer>...</answer>.\n"
#     "The final answer must be valid JSON with exactly one key: \"decision\".\n"

#     "Examples:\n"
#     "Example 1:\n"
#     "<think>1. The target object is the dog lying on the sofa. 2. The red box covers the dog, including its visible body region. 3. Some details are partially ambiguous, but no other visible object in the image matches the question better than the boxed dog. 4. Conclusion: accept</think><answer>{{\"decision\":\"accept\"}}</answer>\n"

#     "Example 2:\n"
#     "<think>1. The target object is the man on the left. 2. The red box covers a different person. 3. The position detail does not match because the boxed person is on the right side, while the question refers to the man on the left. 4. Conclusion: reject</think><answer>{{\"decision\":\"reject\"}}</answer>\n"

#     "Now begin to decide whether the red box involves the referred target object in the question.\n"
# )

# QUERY_REFLECT_REFLECT_RL_PROMPT_TEMPLATE = (
#     "<image>\n"
#     "Question: \"{Question}\"\n"
#     "The image contains a candidate region highlighted by a red box.\n"
#     "Your task is to decide whether the red box matches the referred target object in the question.\n"
#     "Reason step by step:\n"
#     "1. Identify the target object described in the question.\n"
#     "2. Check whether the red box covers the target object or its visible/occluded region, rather than simply judging by the main object inside the box.Accept if the described target cannot be found elsewhere in the image.\n"
#     "3. Check whether the important details in the question, such as relation to other objects, color, position, size and attribute, are also consistent. Reject only if you find a better candidate. Accept if the described target cannot be found elsewhere in the image.\n"
#     "4. Give the final conclusion: \"accept\" or \"reject\". Do not include any explanation. Answer \"accept\" if no other visible object in the image that matches the description of the target object. If the described target cannot be found elsewhere in the image, answer \"accept\". Answer \"reject\" only if you find a better candidate in your reasoning.\n"

#     "Output a concise reasoning process in <think>...</think> and the final answer in <answer>...</answer>.\n"
#     "The final answer must be valid JSON with exactly one key: \"decision\".\n"

#     "Examples:\n"
#     "Example 1:\n"
#     "<think>1. The target object is the man on the left. 2. The red box covers a different person. 3. The position detail does not match because the boxed person is on the right side, while the question refers to the man on the left. 4. Conclusion: reject</think><answer>{{\"decision\":\"reject\"}}</answer>\n"
#     "Example 2:\n"
#     "<think>1. The target object is the dog lying on the sofa. 2. The red box covers the dog, including its visible body region. 3. Some details are partially ambiguous, but no other visible object in the image matches the question better than the boxed dog. 4. Conclusion: accept</think><answer>{{\"decision\":\"accept\"}}</answer>\n"

#     "Now begin to decide whether the red box matches the referred target object in the question.\n"
# )

# "<image>\n"
#     "Question: \"{Question}\"\n"
#     "The image contains a candidate region highlighted by a red box.\n"
#     "Your task is to decide whether the red box matches the referred target object in the question.\n"
#     "Reason step by step:\n"
#     "1. Identify the target object described in the question.\n"
#     "2. Check whether the red box covers the target object or its visible/occluded region, rather than simply judging by the main object inside the box.\n"
#     "3. Check whether the important details in the question, such as relation to other objects, color, position, size and attribute, are also consistent. Reject only if you find a better candidate\n"
#     "4. Give the final conclusion: \"accept\" or \"reject\". Do not include any explanation. Answer \"accept\" if no other visible object in the image that matches the description of the target object.  Answer \"reject\" only if you find a better candidate in your reasoning.\n"

#     "Output a concise reasoning process in <think>...</think> and the final answer in <answer>...</answer>.\n"
#     "The final answer must be valid JSON with exactly one key: \"decision\".\n"

#     "Examples:\n"
#     "<think>1. The target object is the man on the left. 2. The red box covers a different person. 3. The position detail does not match because the boxed person is on the right side, while the question refers to the man on the left. 4. Conclusion: reject</think><answer>{{\"decision\":\"reject\"}}</answer>"
    
#     "Now begin to decide whether the red box matches the referred target object in the question.\n"

# (
#     "<image>\n"
#     "Question: \"{Question}\"\n"
#     "The image contains a candidate region highlighted by a red box.\n"
#     "Your task is to decide whether the red box matches the referred target object in the question.\n"
#     "Reason step by step:\n"
#     "1. Identify the target object described in the question.\n"
#     "2. Identify the main object covered by the red box.\n"
#     "3. Check whether the boxed object's basic category matches the target object. If the main object in the box is not the target object, answer \"reject\".\n"
#     "4. If the basic category matches, check whether the important details in the question, such as color, position, size, attribute, or relation to other objects, are also consistent.\n"
#     "5. Answer \"reject\" if the red box mainly covers the wrong object, only part of the target, or an auxiliary object mentioned only for identification.\n"
#     "6. Answer \"accept\" only if the boxed object matches the target and the key details are broadly consistent.\n"
#     "Output a concise reasoning process in <think>...</think> and the final answer in <answer>...</answer>.\n"
#     "The final answer must be valid JSON with exactly one key: \"decision\".\n"
#     "Examples:\n"
#     "<answer>{{\"decision\":\"accept\"}}</answer>\n"
#     "<answer>{Answer}</answer>\n"
#     "Do not output any other JSON keys."
# )


def build_query_reflect_init_box_prompt(question: str) -> str:
    return QUERY_REFLECT_INIT_BOX_PROMPT_TEMPLATE.format(
        Question=str(question).lower().strip("."),
        Answer=QUERY_REFLECT_INIT_BOX_ANSWER_EXAMPLE,
    )


def build_query_reflect_reflect_prompt(
    question: str,
    previous_think: str,
    previous_answer: str,
    previous_bbox,
    previous_points,
) -> str:
    return QUERY_REFLECT_REFLECT_RL_PROMPT_TEMPLATE.format(
        Question=str(question),
    )


def build_query_reflect_reflect_rl_prompt(question: str) -> str:
    return QUERY_REFLECT_REFLECT_RL_PROMPT_TEMPLATE.format(
        Question=str(question),
    )
