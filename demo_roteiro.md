# Roteiro de Demonstração — musicgen (Mac M4)
**Duração total:** 3 min | **Contexto:** ao vivo após slide 12 ("Let's see it run")

---

## Estratégia

O `pip install` leva ~2 min — não dá para fazer ao vivo. Estratégia:

1. **Dias antes:** setup completo no Mac M4 (seção abaixo)
2. **No palco:** abrir terminal LIMPO → mostrar que o repo está lá → ativar venv → demo
3. A audiência vê "zero a rodando" sem esperar o install

---

## Preparação (fazer dias antes, no Mac M4)

### 1. Dependências do sistema

```bash
# Xcode CLI (se não tiver)
xcode-select --install

# Homebrew (se não tiver)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# FluidSynth
brew install fluidsynth

# Python 3.12 (se não tiver via brew)
brew install python@3.12
```

### 2. Clonar e instalar

```bash
cd ~
git clone https://github.com/dobidu/layered_music_gen.git
cd layered_music_gen

python3 -m venv .venv
source .venv/bin/activate

pip install -e '.[dev]'
pip install psutil               # para benchmarks

# confirmar
musicgen --version
```

### 3. SoundFonts

```bash
# Copiar seus arquivos .sf2 para:
ls sf/
# beat/     → *.sf2
# melody/   → *.sf2
# harmony/  → *.sf2
# bassline/ → *.sf2
```

> **Mínimo:** 1 arquivo por pasta. Mais = melhor variedade tímbrica.

### 4. Gerar dataset de backup (Plano B)

```bash
source ~/.venv/bin/activate 2>/dev/null || source ~/layered_music_gen/.venv/bin/activate
cd ~/layered_music_gen

musicgen generate \
  --count 5 \
  --seed 42 \
  --out /tmp/demo_dataset_backup \
  --genre jazz

# confirmar que gerou
ls /tmp/demo_dataset_backup/
```

### 5. Pré-executar o notebook

```bash
cd ~/layered_music_gen
source .venv/bin/activate
jupyter notebook notebooks/musicgen_demo.ipynb
```

- Executar **Kernel → Run All**
- Confirmar que células 5, 6 e 12 têm saída
- **Deixar o browser aberto** em background

### 6. Verificação final

```bash
cd ~/layered_music_gen && source .venv/bin/activate

musicgen --version                               # deve retornar 0.2.0
musicgen list-genres                             # deve listar 8 gêneros
fluidsynth --version                             # deve retornar versão
pytest -m "not slow" -q --tb=no | tail -2       # deve mostrar 1046 passed
```

---

## Preparação (30 min antes da palestra)

```bash
# Terminal → fonte ≥ 18pt
# Terminal.app: Cmd+, → Profiles → Font → aumentar
# iTerm2: Cmd+, → Profiles → Text → Font size

# Limpar dataset da última vez
rm -rf /tmp/demo_dataset

# Confirmar backup existe
ls /tmp/demo_dataset_backup/000000/sample.json && echo "backup OK"

# Confirmar notebook aberto com saídas visíveis
# (só olhar no browser — não re-executar)

# Deixar DUAS abas prontas no terminal:
#   Aba 1: ~/layered_music_gen (venv ativa) — para o demo
#   Aba 2: ~/layered_music_gen (venv ativa) — reserva
```

---

## DEMO AO VIVO

> **Contexto:** slide 12 na tela. Você abre um terminal novo.

---

### ABERTURA — "Zero a rodando" `[0:00 – 0:30]`

> Abrir terminal limpo. Digitar devagar — audiência precisa ler.

```bash
# mostrar que é um repo real
ls ~/layered_music_gen
```

```bash
cd ~/layered_music_gen
```

```bash
# isso é tudo que precisaria para começar do zero:
# git clone https://github.com/dobidu/layered_music_gen.git
# python3 -m venv .venv
# pip install -e '.[dev]'
# brew install fluidsynth
# (fiz antes — leva ~2 min)

source .venv/bin/activate
```

> **FALA:**  
> "Três comandos — clone, venv, pip install. FluidSynth pelo brew.  
> Fiz antes pra não perder tempo. Agora com o ambiente pronto:"

---

### PARTE A — Terminal `[0:30 – 1:30]`

#### A1 — Listar gêneros `[~15s]`

```bash
musicgen list-genres
```

> **FALA:**  
> "Oito gêneros embutidos — cada um é um arquivo `spec.json` com limites de tempo,  
> swing, pesos de acordes, padrões de bateria, perfil de FX.  
> São composíveis — posso combinar qualquer subconjunto."

---

#### A2 — Gerar com jazz `[~35s]`

```bash
musicgen generate \
  --count 3 \
  --seed 42 \
  --out /tmp/demo_dataset \
  --genre jazz
```

> **FALA (enquanto gera):**  
> "Semente 42, 3 amostras, gênero jazz. Pipeline completo:  
> sampler constrangido pelo GenreSpec → geradores MIDI → FluidSynth →  
> FX pedalboard → anotador → `sample.json` escrito por último."

> **FALA (ao terminar):**  
> "Pronto."

---

#### A3 — Inspecionar saída `[~10s]`

```bash
ls /tmp/demo_dataset/000000/
```

```bash
python3 -c "
import json, pathlib
d = json.loads(pathlib.Path('/tmp/demo_dataset/000000/sample.json').read_text())
print(f'tempo:      {d[\"tempo_bpm\"]} BPM   (jazz: 80–200)')
print(f'swing:      {d[\"swing\"]:.3f}        (jazz: 0.60–0.75)')
print(f'key:        {d[\"key\"]}')
print(f'split:      {d[\"split\"]}')
print(f'musicality: {d[\"musicality\"][\"score\"]:.3f}')
"
```

> **FALA:**  
> "`sample.json` escrito por último — sentinel de completude, usado para retomada.  
> Tempo e swing dentro dos limites do jazz. Split determinístico da semente — sempre igual."

---

### PARTE B — Notebook `[1:30 – 2:30]`

> Alternar para o browser. Notebook já aberto com saídas visíveis.

---

#### B1 — Seção 5: jazz puro `[~20s]`

> Rolar até `## 5. Genre generation — single genre`

> **FALA:**  
> "Carregamos a GenreSpec do jazz diretamente em Python —  
> tempo_min=80, tempo_max=200, swing 0.60–0.75.  
> A amostra gerada cai dentro dos limites — assert passa."

---

#### B2 — Seção 6: composição jazz + latin `[~30s]`

> Rolar até `## 6. Genre composition — jazz + latin`

> **FALA:**  
> "Composição de gêneros — o ponto central do v0.2.  
> `merge_genres([jazz, latin])` calcula automaticamente:"

> Apontar para os valores no output:

> **FALA:**  
> "Intervalos *hard* — tempo e swing — viram interseção.  
> Jazz é 80–200, latin é 90–140: fundido é 90–140.  
> Pesos *soft* — time signature, escalas, tipos de acorde — viram média ponderada.  
> Nenhuma mudança nos geradores. O GenreSpec desce o pipeline inteiro."

---

#### B3 — Seção 12: determinismo `[~20s]`

> Rolar até `## 12. Determinism check`

> **FALA:**  
> "Contrato fundamental. Mesma semente, dois runs independentes —"

> Apontar para os dois hashes e o PASS:

> **FALA:**  
> "SHA-256 idêntico. MIDI e `sample.json` bit a bit iguais sempre.  
> Não importa a máquina — M4 aqui, Ryzen em outro lab, CI na nuvem."

---

### PARTE C — Fechar `[2:30 – 3:00]`

> Voltar ao terminal.

```bash
pytest -m "not slow" -q --tb=no 2>&1 | tail -2
```

> **FALA:**  
> "1046 testes em menos de 4 segundos — sem FluidSynth, sem sf2.  
> Repositório: `github.com/dobidu/layered_music_gen`, tag `v0.2.0`.  
> Obrigado — perguntas?"

---

## Mapa de tempo

| Relógio | Ação | Local |
|---|---|---|
| `0:00` | `ls ~/layered_music_gen` → `cd` → `source .venv/bin/activate` | Terminal |
| `0:20` | Mostrar linha de clone (não executar) | Terminal |
| `0:30` | `musicgen list-genres` | Terminal |
| `0:45` | `musicgen generate --count 3 --seed 42 --genre jazz` | Terminal |
| `1:20` | `ls 000000/` + python3 inspecionar sample.json | Terminal |
| `1:30` | Seção 5 (jazz spec + assert) | Notebook |
| `1:50` | Seção 6 (merge jazz + latin) | Notebook |
| `2:20` | Seção 12 (SHA-256 PASS) | Notebook |
| `2:30` | `pytest -m "not slow"` | Terminal |
| `2:50` | Encerrar | — |

---

## Plano B

| Problema | Ação imediata |
|---|---|
| `generate` demora mais que 40s | `Ctrl+C` → `ls /tmp/demo_dataset_backup/000000/` → seguir script normalmente com o backup |
| FluidSynth erro / crash | `musicgen generate --count 1 --seed 42 --out /tmp/d --genre jazz --output-mode midi-only` (sem síntese) → mostra MIDI + sample.json |
| Notebook não carrega | No terminal: `python3 -c "from musicgen.genre import load_genre, merge_genres; j=load_genre('jazz','genres'); l=load_genre('latin','genres'); m=merge_genres([j,l]); print(f'merged tempo: [{m.tempo_min}, {m.tempo_max}]')"` |
| `pytest` falha | `cat benchmarks/results/*.json \| python3 -m json.tool \| grep -E '"mean_ms"|"n"'` — mostra números do benchmark |
| Sem internet (clone) | `ls ~/layered_music_gen` + `git log --oneline -3` — prova que o repo é real |

---

## Checklist — 30 min antes

```
□ Mac M4 na tomada / bateria cheia
□ Modo Não Perturbe ativado
□ Terminal aberto em ~/layered_music_gen, fonte ≥ 18pt, fundo escuro
□ venv ativa: which python → .../layered_music_gen/.venv/bin/python
□ musicgen --version → 0.2.0
□ fluidsynth --version → qualquer versão
□ sf2 confirmados: ls sf/beat/ sf/melody/ sf/harmony/ sf/bassline/
□ /tmp/demo_dataset removido: rm -rf /tmp/demo_dataset
□ /tmp/demo_dataset_backup/000000/sample.json existe
□ Notebook aberto no browser, células 5/6/12 com saída visível (não re-executar)
□ pytest OK: pytest -m "not slow" -q --tb=no | tail -1 → "1046 passed"
□ Ensaio completo cronometrado (alvo: < 2m45s para sobrar margem)
```

---

## Notas para ensaio

- **Falar devagar** ao digitar — audiência lê mais devagar que você digita
- **Não esperar silêncio** durante o `generate` — falar o pipeline enquanto roda
- **Seção 6 é o pico** — dar tempo para a audiência ler os valores de interseção antes de explicar
- Se sobrar tempo: na seção 12, rolar para cima e mostrar o código (`sha256_file`) antes de mostrar o resultado
- Alvo real: terminar em **2m40s** — 20s de buffer para perguntas durante o demo
