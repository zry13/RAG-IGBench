# RAG-IGBench: Innovative Evaluation for RAG-based Interleaved Generation in Open-domain Question Answering
In real-world scenarios, providing user queries with visually enhanced responses can considerably benefit understanding and memory, underscoring the great value of interleaved image-text generation. Despite recent progress like the visual autoregressive model which unifies both text and image processing in single transformer architecture, generating high-quality interleaved content remains challenging. Moreover, evaluations of these interleaved sequences largely remain underexplored, with existing benchmarks often limited by unimodal metrics that inadequately assess the intricacies of combined image-text outputs. 

Therefore, we present **RAG-IGBench**, a thorough benchmark designed specifically to evaluate the task of **Interleaved Generation** based on **Retrieval-Augmented Generation** (RAG-IG) in open-domain question answering.

![](assets/case example.jpg)

![](assets/overview.jpg)


## 📚 Setup
We have host the RAG-IGBench dataset on Huggingface, where you should request access on this page first and will be automatically approved. The format of `data.json` is:
```
[
    {
        "id": "a number",
        "query": "user question",
        "documents": [
            "content of doc#1",
            "content of doc#2",
            ...
        ],
        "images":[
            [IMG#1, IMG#2, ...],     # images of doc#1
            [IMG#x, IMG#x+1, ...],      # images of doc#2
            ...
        ],
        "gt_answer": "answer str",
        "category":[
            "Finance",      # topic of the query
            "what-is"       # one of "what-is", "how-to", "yes-or-no" and "head-to-head"
        ]
    }
]
```
You can download the `images.zip` to get all images. The images file structure is :
```
INPUT_DIR
    |INPUT_FILE(data.json)
    |images
        |1                              # images folder for query 1
            |1                              # images folder for doc#1
                |IMG#1.jpg
                |IMG#2.jpg
                |...
            |2                              # images folder for doc#2
                |IMG#x.jpg
                |...
            |3                              # images folder for doc#3
        |2                              # images folder for query 2
            |...
        |3                              # images folder for query 3
            |...
        |..
```
## 🛠️ Usage
### Answer generation
We have to modify the inference code for various models to support interleaved multi-images inputs. We have prepared some testing scripts for currently available open-source and closed-source models in the `./model_generation` folder. The models we support now include **GPT4o**, **Claude**, **Gemini**, **LLaVa-OV**, **InternVL2.5**, **Qwen2/2.5VL**, **NVLM_D** and **Deepseekvl2**. For the evaluation of these models, you only need to modify the model's path and the path to RAG-IGbench's dataset to generate answers. The test result will be saved in a JSONL file.
### Evaluation
Use the `./eval.py` script to score the model's responses. What you need to provide is:
```
output_file = ''              # Save the evaluation result
input_file = ''               # Model answer file
clip_model_path = ''          # The model used to calculate Clip-Score, default as CLIP-ViT-Large-Patch14 from OpenAi
embedding_model_path = ''     # The model used to calculate Alignment-Score, default as Conan-embedding
```
Finally, the evaluation result will be saved in a JSONL file and the final scores will be printed in stdout, or you can run `./scores.py` with evaluation result file as input to get the final scores.
