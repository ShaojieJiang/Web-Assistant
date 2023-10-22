# Web Assistant: An open version of WebGPT

Reproducing the WebGPT[^webgpt] work using `gpt-3.5-turbo` api calls and DuckDuckGo search.
Enables votting for data annotation.
Implemented using LangChain, Streamlit (for deployment), and Gradio (for fast prototyping).

[^webgpt]: [Nakano, Reiichiro, et al. "Webgpt: Browser-assisted question-answering with human feedback." arXiv preprint arXiv:2112.09332 (2021).](https://arxiv.org/abs/2112.09332)

Example usages:
* Set it up for as personal web copilot, without subscribing ChatGPT Plus
* Annotate comparisons data using flagging for RM training, which is essential for RLHF
* Used as a prototype for designing your RM data collection interface

<!-- The annotated data will be written to your local path under `flagged` -->

## Setup

After cloning this repo to your machine, run the following commands in your shell:
```shell
pip install -r requirements.txt
export OPENAI_API_KEY=you_openai_api_key
```

## Start

Simply run this in your terminal:
```shell
streamlit run web_assistant.py
```

<!-- ## Sharing with your friends or family

If you want to share the tool running on your machine, you can change `share: false` to `share: true` in `config.yaml`.
Then after restarting the program, you will see a temporary link hosted by Gradio that can be accessed by others.
> _NOTE:_ Sharing this link will allow others to use the tool that costs your OpenAI credits, please be aware of that. -->

## TODO's

- [ ] Query rewriting in-between the latest message and response (making use of chat history)
- [ ] Simplify chain calls and prompts
- [ ] Add references
- [ ] Add crawled webpages for debugging
- [ ] Summarise webpages using models/OpenAI api
- [ ] Deployment of open-source models
