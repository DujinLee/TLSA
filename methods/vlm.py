"""Generative VLM label discovery (Step 1 of TLSA).

Every target image is queried with ``--num_prompts`` semantically equivalent
but lexically diverse questions; the final discovered label is the majority
vote over the answers.  Two backends are provided:

* ``blip``  -- BLIP-VQA (``blip_vqa`` / ``vqav2``) from the LAVIS library.
                This is the backend used for all main results in the paper.
* ``qwen``  -- any OpenAI-compatible chat endpoint serving a Qwen3-VL model
                (we use vLLM).  Used for the VLM-robustness study.

Both backends share the same rule-based answer post-processing, so the rest of
the pipeline is unaware of which VLM produced a label.
"""

import asyncio
import base64
from io import BytesIO

import torch

# Answers that do not name an object category are discarded.
COLORS = ['green', 'pink', 'blue', 'red', 'yellow',
          'black', 'purple', 'white', 'brown', 'silver']
PREPOSITIONS = ['on', 'in', 'under', 'with', 'and']
NON_OBJECT_ANSWERS = ["no", "no idea", "yes", "i don't know", "i can't tell"]


def postprocess_answer(answer):
    """Return ``answer`` if it plausibly names an object category, else ``None``.

    Removes non-answers ("yes"/"no"/"i don't know"), possessive or meta
    responses (containing "object" or an apostrophe) and attribute-style
    answers that mention a colour or a preposition.
    """
    if answer is None:
        return None
    answer = answer.strip().lower()
    if not answer:
        return None
    if ("object" in answer) or ("'" in answer) or (answer in NON_OBJECT_ANSWERS):
        return None
    if any(word in COLORS + PREPOSITIONS for word in answer.split()):
        return None
    return answer


class BlipLabelGenerator:
    """BLIP-VQA label generator (LAVIS ``blip_vqa`` pretrained on VQAv2)."""

    def __init__(self, device, max_answer_length=10, num_beams=3):
        from lavis.models import load_model_and_preprocess

        self.device = device
        self.num_beams = num_beams
        self.max_answer_length = max_answer_length
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name="blip_vqa", model_type="vqav2", is_eval=True, device=device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    def preprocess(self, pil_images):
        return [self.vis_processors["eval"](img).unsqueeze(0).to(self.device)
                for img in pil_images]

    @torch.no_grad()
    def generate(self, pil_images, questions):
        """Return a ``[len(questions)][len(pil_images)]`` matrix of answers."""
        model, device = self.model, self.device
        images = self.preprocess(pil_images)
        vision_outputs = model.visual_encoder.forward_features(torch.cat(images))

        answer_list = []
        for question in questions:
            question = self.txt_processors['eval'](question)
            tokenized = model.tokenizer(question,
                                        padding="longest",
                                        truncation=True,
                                        max_length=model.max_txt_len,
                                        return_tensors="pt").to(device)
            tokenized.input_ids[:, 0] = model.tokenizer.enc_token_id

            question_outputs = model.text_encoder.forward_automask(
                tokenized_text=tokenized, visual_embeds=vision_outputs)
            question_states = question_outputs.last_hidden_state.repeat_interleave(
                repeats=self.num_beams, dim=0)
            question_atts = torch.ones(question_states.size()[:-1],
                                       dtype=torch.long).to(device)

            bsz = vision_outputs.size(0)
            bos_ids = torch.full((bsz, 1),
                                 fill_value=model.tokenizer.bos_token_id,
                                 device=device)
            outputs = model.text_decoder.generate(
                input_ids=bos_ids,
                max_length=self.max_answer_length,
                min_length=1,
                num_beams=self.num_beams,
                eos_token_id=model.tokenizer.sep_token_id,
                pad_token_id=model.tokenizer.pad_token_id,
                encoder_hidden_states=question_states,
                encoder_attention_mask=question_atts,
            )
            answer_list.append([
                postprocess_answer(model.tokenizer.decode(o, skip_special_tokens=True))
                for o in outputs
            ])
        return answer_list


class VLLMLabelGenerator:
    """Label generator backed by an OpenAI-compatible chat endpoint (vLLM).

    Used for the Qwen3-VL experiments.  The server must already be running;
    see the README for the launch command.
    """

    SUFFIX = " Answer in a single word or short noun phrase."

    def __init__(self, api_url, model_name, max_concurrency=32,
                 max_tokens=10, max_retries=5, base_delay=1.0):
        self.api_url = api_url
        self.model_name = model_name
        self.max_concurrency = max_concurrency
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.base_delay = base_delay

    @staticmethod
    def _encode(img):
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def generate(self, pil_images, questions):
        return asyncio.run(self._generate(pil_images, questions))

    async def _generate(self, pil_images, questions):
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key="EMPTY", base_url=self.api_url)
        b64_images = [self._encode(img) for img in pil_images]

        async def fetch(q_idx, img_idx, question, b64):
            for attempt in range(self.max_retries):
                try:
                    res = await client.chat.completions.create(
                        model=self.model_name,
                        messages=[{
                            "role": "user",
                            "content": [
                                {"type": "text", "text": question + self.SUFFIX},
                                {"type": "image_url",
                                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
                            ],
                        }],
                        max_tokens=self.max_tokens,
                        temperature=0.0,
                    )
                    return q_idx, img_idx, res.choices[0].message.content
                except Exception as err:  # transient server-side contention
                    msg = str(err).lower()
                    transient = any(s in msg for s in
                                    ("already borrowed", "400", "429", "timeout"))
                    if not transient:
                        print(f"VLM fatal error at [q:{q_idx}, img:{img_idx}]: {err}")
                        return q_idx, img_idx, None
                    await asyncio.sleep(self.base_delay * (2 ** attempt))
            print(f"VLM max retries exceeded for [q:{q_idx}, img:{img_idx}]")
            return q_idx, img_idx, None

        sem = asyncio.Semaphore(self.max_concurrency)

        async def guarded(*args):
            async with sem:
                return await fetch(*args)

        tasks = [guarded(q_idx, img_idx, q, b64)
                 for q_idx, q in enumerate(questions)
                 for img_idx, b64 in enumerate(b64_images)]
        results = await asyncio.gather(*tasks)
        await client.close()

        answer_matrix = [[None] * len(pil_images) for _ in questions]
        for q_idx, img_idx, answer in results:
            answer_matrix[q_idx][img_idx] = postprocess_answer(answer)
        return answer_matrix


def build_label_generator(cfg, device):
    """Instantiate the generative VLM selected by ``--vlm``."""
    if cfg.vlm == 'blip':
        return BlipLabelGenerator(device)
    if cfg.vlm == 'qwen':
        return VLLMLabelGenerator(api_url=cfg.vllm_api_url,
                                  model_name=cfg.vllm_model_name,
                                  max_concurrency=cfg.vlm_max_concurrency)
    raise ValueError(f'unknown --vlm {cfg.vlm}')
