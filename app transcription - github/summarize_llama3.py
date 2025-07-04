from llama_cpp import Llama
import sys
import math

MODEL_PATH = "models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
N_CTX = 32768
CHUNK_TOKENS = 20000  # limite par chunk
SUMMARY_TOKENS = 512

def get_num_tokens(llm, text):
    return len(llm.tokenize(text.encode("utf-8"), add_bos=True))

def build_prompt(content):
    return (
        f"Voici une transcription audio brute :\n{content}\n\n"
        "Tu es un assistant qui répond toujours uniquement en français, sans jamais donner de traduction ou de version anglaise. "
        "Résume ce texte en français, de manière structurée, en listant les points importants, décisions, actions à retenir, sous forme de liste à puces claire et concise."
    )

def summarize_chunk(llm, chunk, print_out=True):
    prompt = build_prompt(chunk)
    if print_out:
        print(f"Chunk de {len(chunk)} caractères. Génération résumé...")
    output = llm(
        prompt,
        max_tokens=SUMMARY_TOKENS,
        stop=["</s>"]
    )
    summary = output["choices"][0]["text"].strip()
    if print_out:
        print("\n--- Résumé chunk ---\n")
        print(summary)
    return summary

def chunk_text_by_tokens(llm, text, max_tokens):
    # Coupe le texte en paquets de max_tokens tokens sans casser des phrases
    sentences = text.split('\n')  # on coupe d'abord par lignes, c'est safe pour une transcription
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if not sentence.strip():
            continue
        tmp_chunk = current_chunk + sentence + "\n"
        if get_num_tokens(llm, build_prompt(tmp_chunk)) > max_tokens:
            # Si en ajoutant la phrase, on dépasse, on bloque ici
            chunks.append(current_chunk.strip())
            current_chunk = sentence + "\n"
        else:
            current_chunk = tmp_chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks

def summarize(file_path):
    # Instancie le modèle une seule fois
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        verbose=True
    )

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    prompt = build_prompt(content)
    num_tokens = get_num_tokens(llm, prompt)
    print(f"Nombre de tokens du prompt : {num_tokens}")

    if num_tokens + SUMMARY_TOKENS <= N_CTX and num_tokens <= CHUNK_TOKENS:
        # Cas classique : on fait un résumé direct
        print("⏳ Génération résumé global...")
        output = llm(
            prompt,
            max_tokens=SUMMARY_TOKENS,
            stop=["</s>"]
        )
        print("\n--- Résumé généré ---\n")
        print(output["choices"][0]["text"].strip())
    else:
        # Cas trop gros : on découpe
        print("Le texte est trop long, découpage en paquets de 20 000 tokens...")
        chunks = chunk_text_by_tokens(llm, content, CHUNK_TOKENS)
        all_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"\n⏳ Chunk {i+1}/{len(chunks)}")
            summary = summarize_chunk(llm, chunk, print_out=False)
            print(f"Résumé du chunk {i+1} :\n{summary}\n")
            all_summaries.append(summary)
        print("⏳ Génération du méta-résumé final...")
        meta_input = "\n\n".join([f"Résumé {i+1} :\n{summary}" for i, summary in enumerate(all_summaries)])
        meta_prompt = (
            "Voici plusieurs résumés partiels d'une longue transcription audio.\n\n"
            f"{meta_input}\n\n"
            "Fais une synthèse globale, uniquement en français, des points clés à retenir, décisions, actions, sous forme de liste à puces concise."
        )
        meta_output = llm(
            meta_prompt,
            max_tokens=SUMMARY_TOKENS,
            stop=["</s>"]
        )
        print("\n--- MÉTA-RÉSUMÉ GÉNÉRÉ ---\n")
        print(meta_output["choices"][0]["text"].strip())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python summarize_llama3.py <fichier_transcription.txt>")
        sys.exit(1)
    summarize(sys.argv[1])
