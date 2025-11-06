"""Normalizações e regras específicas para termos médicos."""

from functools import lru_cache

# Dicionário de correções de termos médicos comuns
MEDICAL_TERMS: dict[str, str] = {
    "hemoglohina": "hemoglobina",
    "glocose": "glicose",
    "creatnina": "creatinina",
    "hemacias": "hemácias",
    "leucocitos": "leucócitos",
    "plaquetas": "plaquetas",
    "hematocrito": "hematócrito",
    "vcm": "VCM",
    "hcm": "HCM",
    "chcm": "CHCM",
    "rdw": "RDW",
}


@lru_cache(maxsize=128)
def normalize_medical_term(term: str) -> str:
    """
    Normaliza um termo médico comum.

    Args:
        term: Termo a normalizar

    Returns:
        Termo normalizado
    """
    term_lower = term.lower().strip()
    return MEDICAL_TERMS.get(term_lower, term)


def normalize_medical_text(text: str) -> str:
    """
    Normaliza texto médico completo, corrigindo termos comuns.

    Args:
        text: Texto a normalizar

    Returns:
        Texto normalizado
    """
    if not text:
        return text

    words = text.split()
    normalized_words = [normalize_medical_term(word) for word in words]
    return " ".join(normalized_words)


def standardize_decimal(text: str, target_separator: str = ".") -> str:
    """
    Padroniza separador decimal.

    Args:
        text: Texto com número decimal
        target_separator: Separador desejado ('.' ou ',')

    Returns:
        Texto com separador padronizado
    """
    if not text:
        return text

    if target_separator == ".":
        # Substitui vírgula por ponto
        return text.replace(",", ".")
    if target_separator == ",":
        # Substitui ponto por vírgula (cuidado com números inteiros)
        # Só substitui se houver dígitos antes e depois
        import re

        return re.sub(r"(\d+)\.(\d+)", r"\1,\2", text)

    return text


def clean_ocr_artifacts(text: str) -> str:
    """
    Remove artefatos comuns do OCR.

    Args:
        text: Texto com artefatos

    Returns:
        Texto limpo
    """
    if not text:
        return text

    # Remove caracteres problemáticos
    text = text.replace("|", "").replace("—", "-").replace("_", " ")
    # Normaliza espaços múltiplos
    text = " ".join(text.split())
    return text.strip()


def remove_label_from_text(text: str, label: str) -> str:
    """
    Remove a label do texto extraído pelo OCR de forma robusta.

    Remove todas as palavras da label do texto, incluindo variações parciais,
    palavras em ordem diferente, e padrões comuns do OCR.

    Args:
        text: Texto extraído pelo OCR (pode conter a label)
        label: Label do campo a ser removida

    Returns:
        Texto sem a label
    """
    if not text or not label:
        return text

    import re

    # Normaliza espaços
    text = " ".join(text.split())
    label_normalized = " ".join(label.split())
    original_text = text

    # Remove padrões comuns do final primeiro (para evitar conflitos)
    # Remove padrões como "(Código / Nome)", "ncionário (Código / Nome)", etc.
    final_patterns = [
        r"\s*\([^)]*código[^)]*\)",  # (Código / Nome)
        r"\s*\([^)]*nome[^)]*\)",  # (Nome)
        r"\s*ncionário\s*\([^)]*\)",  # ncionário (Código / Nome)
        r"\s*funcionário\s*\([^)]*\)",  # funcionário (Código / Nome)
        r"\s*\([^)]*código\s*/\s*nome[^)]*\)",  # (Código / Nome) com espaços
    ]

    for pattern in final_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Separa palavras da label
    label_words = label_normalized.split()
    label_words_lower = [w.lower() for w in label_words]
    
    # Padrões para remover do início (label completa e variações)
    label_patterns = []
    
    # Label completa com variações de case
    label_escaped = re.escape(label_normalized)
    label_patterns.extend([
        rf"^{label_escaped}\s*:?\s*",
        rf"^{re.escape(label_normalized.lower())}\s*:?\s*",
        rf"^{re.escape(label_normalized.upper())}\s*:?\s*",
        rf"^{re.escape(label_normalized.title())}\s*:?\s*",
    ])
    
    # Para labels com múltiplas palavras, cria padrões para subconjuntos
    if len(label_words) > 1:
        # Primeira palavra
        first_word = label_words[0]
        label_patterns.append(rf"^{re.escape(first_word)}\s+")
        
        # Primeiras duas palavras
        if len(label_words) >= 2:
            first_two = " ".join(label_words[:2])
            label_patterns.append(rf"^{re.escape(first_two)}\s*:?\s*")
        
        # Última palavra (para casos como "Data da Ficha" → "Ficha")
        last_word = label_words[-1]
        label_patterns.append(rf"^{re.escape(last_word)}\s+")
        
        # Todas as palavras exceto artigos/preposições
        # Remove artigos e preposições comuns
        stop_words = {"da", "de", "do", "das", "dos", "a", "o", "e", "em", "na", "no"}
        content_words = [w for w in label_words if w.lower() not in stop_words]
        if len(content_words) > 1:
            content_label = " ".join(content_words)
            label_patterns.append(rf"^{re.escape(content_label)}\s*:?\s*")

    # Adiciona variações específicas para campos comuns
    label_lower = label_normalized.lower()
    if "data" in label_lower and "ficha" in label_lower:
        label_patterns.extend([
            r"^data\s+ficha\s*:?\s*",
            r"^data\s+da\s+ficha\s*:?\s*",
            r"^ficha\s+",  # Remove "Ficha" do início
        ])
    
    if "funcionario" in label_lower or "funcionário" in label_lower:
        # Remove variações parciais de "funcionário" apenas do início
        label_patterns.extend([
            r"^ncionário\s+",  # Remove "ncionário" do início (parte de "funcionário")
            r"^ncionario\s+",
            r"^funcionário\s+",
            r"^funcionario\s+",
        ])

    # Remove padrões do início
    for pattern in label_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Abordagem mais agressiva: remove palavras da label do início do texto
    # Isso é mais seguro que remover de qualquer posição
    text_words = text.split()
    words_to_remove = set()
    
    # Identifica palavras do início que correspondem à label (case-insensitive)
    # Só remove do início para evitar remover valores válidos
    for i, word in enumerate(text_words):
        if i >= len(label_words) * 2:  # Limita busca às primeiras palavras
            break
            
        word_lower = word.lower().rstrip(".,:;!?")
        # Verifica se a palavra corresponde a alguma palavra da label
        for label_word in label_words_lower:
            # Match exato
            if word_lower == label_word:
                words_to_remove.add(i)
                break
            # Match parcial apenas para casos conhecidos (ex: "ncionário" de "funcionário")
            elif label_word == "funcionário" or label_word == "funcionario":
                if word_lower in ["ncionário", "ncionario", "funcionário", "funcionario"]:
                    words_to_remove.add(i)
                    break
    
    # Remove palavras identificadas
    text_words = [w for i, w in enumerate(text_words) if i not in words_to_remove]
    text = " ".join(text_words)

    # Remove padrões específicos conhecidos que podem aparecer
    # Remove "Ficha" quando a label contém "Data da Ficha"
    if "ficha" in label_lower:
        text = re.sub(r"^ficha\s+", "", text, flags=re.IGNORECASE)
    
    # Remove "Sexo" quando a label é "Sexo"
    if label_lower == "sexo":
        text = re.sub(r"^sexo\s+", "", text, flags=re.IGNORECASE)
    
    # Remove "Nome" quando a label contém "Nome"
    if "nome" in label_lower:
        text = re.sub(r"^nome\s+", "", text, flags=re.IGNORECASE)
    
    # Remove "Empresa" quando a label é "Empresa"
    if label_lower == "empresa":
        text = re.sub(r"^empresa\s+", "", text, flags=re.IGNORECASE)
        # Remove "presa" do final (artefato comum do OCR quando lê "Empresa" mal)
        text = re.sub(r"\s+presa\s*$", "", text, flags=re.IGNORECASE)
    
    # Remove "ncionário" quando a label contém "funcionario" (pode aparecer no meio do texto)
    # Isso é um artefato comum do OCR quando lê "Funcionario" mal
    if "funcionario" in label_lower or "funcionário" in label_lower:
        # Remove "ncionário" apenas se estiver isolado (não parte de outra palavra)
        text = re.sub(r"\s+ncionário\s+", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+ncionario\s+", " ", text, flags=re.IGNORECASE)
        # Remove do final também
        text = re.sub(r"\s+ncionário\s*$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+ncionario\s*$", "", text, flags=re.IGNORECASE)

    # Remove espaços extras e limpa
    text = " ".join(text.split())
    return text.strip()


