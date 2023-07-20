import time
from functools import partial

import gradio as gr
import hydra
import openai
from duckduckgo_search import DDGS
from omegaconf import DictConfig


def search_res_prompt(i, url, body):
    return f"""link [{i+1}]: {url}
content: {body}

"""


def clear_fn():
    # in_box, out_box, comparison_section, serp_box, voting_btns, alt_btns, state
    return (
        "",
        "",
        gr.update(visible=False),
        "",
        gr.update(visible=False),
        gr.update(visible=False),
        None,
    )


def render_flag_btns(flagging_options):
    return [
        (
            gr.Button("Flag as " + flag_option),
            flag_option,
        )
        for flag_option in flagging_options
    ]


def prepare_openai_input(last_message):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    # get search results
    start = time.time()
    ddgs = DDGS()
    results = ddgs.text(last_message)
    end = time.time()
    print(f"Time on search: {end-start} seconds")

    search_results = ""
    for i, result in enumerate(results):
        result_item = search_res_prompt(i, result["href"], result["body"])
        search_results += result_item
        if i > 5:
            break
    message_with_search = (
        f"{last_message}\nBelow is what I found on the web:\n{search_results}"
    )

    messages.append({"role": "user", "content": message_with_search})

    return messages, search_results


def get_openai_output(message, state: dict, api_key=None):
    openai.api_key = api_key
    if state is None:
        state = {"history": []}
    history: list = state.get("history", [])

    if message:
        messages, search_results = prepare_openai_input(message)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
        )
        response = completion["choices"][0]["message"]["content"]
        history.append((message, response))

        state["history"] = history
        state["messages"] = messages

    # in_box, out_box, comparison_section, serp_box, voting_btns, alt_btns, state
    return (
        "",
        history,
        gr.update(visible=False),
        search_results,
        gr.update(visible=True),
        gr.update(visible=True),
        state,
    )


def get_alternative_openai_outputs(state: dict, api_key=None, n=5):
    openai.api_key = api_key
    messages = state.get("messages", [])

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        n=n,
    )

    choice_inds = [str(i) for i in range(1, len(completion["choices"]) + 1)]
    result_str = "\n\n".join(
        [
            f"Response {i+1}:\n{choice['message']['content']}"
            for i, choice in enumerate(completion["choices"])
        ]
    )

    # comparison_section, comparisons, alt_box
    return (
        gr.update(visible=True),
        gr.update(choices=choice_inds, value=[]),
        gr.update(value=result_str),
    )


class FlagMethod:
    def __init__(self, flagging_callback, flag_option=None):
        self.flagging_callback = flagging_callback
        self.flag_option = flag_option
        self.__name__ = "Flag"

    def __call__(self, *flag_data):
        self.flagging_callback.flag(flag_data, flag_option=self.flag_option)
        return gr.update(visible=False)


@hydra.main(config_path="./", config_name="config", version_base="1.2.0")
def main(cfg: DictConfig):
    answer_fn = partial(get_openai_output, api_key=cfg.openai_api_key)
    alt_response_fn = partial(
        get_alternative_openai_outputs, api_key=cfg.openai_api_key, n=5
    )

    demo = gr.Blocks(css="#chat_box {white-space: pre-line}")

    with demo:
        state = gr.State(None)

        out_box = gr.Chatbot(label="History", elem_id="chat_box")
        in_box = gr.Textbox(label="Your message:")

        voting_btns = gr.Row(visible=False)
        with voting_btns:
            flag_btns = render_flag_btns(flagging_options=["upvote", "downvote"])

        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear all")

        alt_btns = gr.Row(visible=False)
        with alt_btns:
            alt_res_btn = gr.Button("Alternative responses")

        comparison_section = gr.Column(visible=False)
        with comparison_section:
            with gr.Row():
                comparisons = gr.Dropdown(
                    label="Order candidate responses from BEST to WORST."
                    " If you can't distinguish some, just keep one.",
                    multiselect=True,
                    interactive=True,
                )
                flag_comp_btn = gr.Button("Submit comparisons")
            alt_box = gr.Textbox(label="Alternative responses:")

        serp_box = gr.Textbox(label="Search results", interactive=False)

        # actions
        flag_components = [out_box, serp_box]
        output_components = [
            in_box,
            out_box,
            comparison_section,
            serp_box,
            voting_btns,
            alt_btns,
            state,
        ]

        submit_kwargs = {
            "fn": answer_fn,
            "inputs": [in_box, state],
            "outputs": output_components,
            "show_progress": True,
        }

        in_box.submit(**submit_kwargs)
        submit_btn.click(**submit_kwargs)
        clear_btn.click(fn=clear_fn, inputs=[], outputs=output_components)
        alt_res_btn.click(
            fn=alt_response_fn,
            inputs=[state],
            outputs=[comparison_section, comparisons, alt_box],
            show_progress=True,
        )

        voting_callback = gr.CSVLogger()
        voting_callback.setup(flag_components, flagging_dir="flagged/voting")
        for flag_btn, flag_option in flag_btns:
            flag_method = FlagMethod(voting_callback, flag_option)
            flag_btn.click(
                flag_method,
                inputs=flag_components,
                outputs=[voting_btns],
                preprocess=False,
                queue=False,
            )

        comparison_callback = gr.CSVLogger()
        comparison_components = flag_components + [alt_box, comparisons]
        comparison_callback.setup(
            comparison_components, flagging_dir="flagged/comparison"
        )
        comparison_method = FlagMethod(comparison_callback, flag_option="comparisons")
        flag_comp_btn.click(
            fn=comparison_method,
            inputs=comparison_components,
            outputs=[comparison_section],
            preprocess=False,
        )

    demo.launch(share=cfg.share)


if __name__ == "__main__":
    main()
