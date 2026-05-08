# Roteiro de Demonstração — musicgen
**Duração total:** 3 min | **Slides:** encerram no slide 12 ("Let's see it run")

---

## Preparação (fazer ANTES da apresentação)

```bash
# 1. Ativar venv
cd ~/musicgen
source .venv/bin/activate

# 2. Limpar dataset anterior
rm -rf /tmp/demo_dataset

# 3. Abrir notebook (segunda tela ou segunda aba)
jupyter notebook notebooks/musicgen_demo.ipynb
#    → pré-executar células 5, 6, 12 (Kernel → Run All)
#    → deixar as saídas visíveis, não limpar

# 4. Aumentar fonte do terminal: mínimo 18pt
# 5. Testar lista de gêneros:
musicgen list-genres
```

---

## PARTE A — Terminal `[0:00 – 1:00]`

### A1 — Mostrar os gêneros disponíveis `[~15s]`

```bash
musicgen list-genres
```

> **FALA:**  
> "O musicgen vem com 8 gêneros embutidos — cada um define limites de tempo,  
> swing, padrões de bateria e pesos de acordes. Vou gerar algumas amostras com jazz."

---

### A2 — Gerar 3 amostras com gênero `[~30s]`

```bash
musicgen generate \
  --count 3 \
  --seed 42 \
  --out /tmp/demo_dataset \
  --genre jazz
```

> **FALA (enquanto gera):**  
> "Seed 42, 3 amostras, gênero jazz. O pipeline completo: amostrador,  
> geradores MIDI, FluidSynth, FX, anotador — tudo controlado pela GenreSpec do jazz."

> **FALA (quando terminar):**  
> "Pronto. Vamos ver o que foi criado."

---

### A3 — Inspecionar a saída `[~15s]`

```bash
ls /tmp/demo_dataset/000000/
```

```bash
python3 -c "
import json, pathlib
d = json.loads(pathlib.Path('/tmp/demo_dataset/000000/sample.json').read_text())
print(f\"tempo:      {d['tempo_bpm']} BPM\")
print(f\"swing:      {d['swing']:.3f}\")
print(f\"key:        {d['key']}\")
print(f\"split:      {d['split']}\")
print(f\"musicality: {d['musicality']['score']:.3f}\")
"
```

> **FALA:**  
> "O `sample.json` foi escrito por último — é o sentinel de completude.  
> Tempo dentro dos limites do jazz: 80–200 BPM. Swing entre 0.60 e 0.75.  
> Split determinístico: 'train', 'valid' ou 'test' — sempre o mesmo para essa semente."

---

## PARTE B — Notebook `[1:00 – 2:30]`

> Alternar para o browser com o notebook já aberto e com saídas visíveis.

---

### B1 — Seção 5: geração com jazz `[~25s]`

> Rolar até **`## 5. Genre generation — single genre`**, destacar a saída.

> **FALA:**  
> "Aqui carregamos a GenreSpec do jazz diretamente — podemos ver os limites:  
> tempo_min=80, tempo_max=200, swing entre 0.60 e 0.75.  
> A amostra gerada respeita esses limites — vemos o assert passando."

---

### B2 — Seção 6: composição de gêneros `[~35s]`

> Rolar até **`## 6. Genre composition — jazz + latin`**, destacar a saída.

> **FALA:**  
> "Agora o mais interessante: composição de gêneros.  
> Passamos `--genre jazz --genre latin` — o `merge_genres` calcula  
> automaticamente a interseção dos intervalos hard e a média ponderada  
> dos pesos soft."

> Apontar para os valores no output:

> **FALA:**  
> "O intervalo de tempo fundido é a interseção: se jazz é 80–200 e latin é 90–140,  
> o resultado é 90–140. Nenhuma mudança de código nos geradores —  
> a GenreSpec descida para o sampler resolve tudo."

---

### B3 — Seção 12: verificação de determinismo `[~30s]`

> Rolar até **`## 12. Determinism check`**, destacar a saída.

> **FALA:**  
> "Por fim, o contrato fundamental do musicgen.  
> Mesma semente, duas execuções completamente independentes —  
> SHA-256 idêntico no `sample.json`."

> Apontar para o "PASS":

> **FALA:**  
> "PASS. Não importa a máquina, não importa quando você roda —  
> MIDI e metadados são bit a bit idênticos. Isso é o que torna  
> o dataset reproduzível para a comunidade."

---

## PARTE C — Encerramento `[2:30 – 3:00]`

> Voltar ao terminal.

```bash
pytest -m "not slow" -q --tb=no 2>&1 | tail -2
```

> **FALA:**  
> "1046 testes, menos de 4 segundos — sem FluidSynth.  
> O repositório está em github.com/dobidu/layered_music_gen, tag v0.2.0.  
> Obrigado — perguntas?"

---

## Mapa de tempo

| Momento | Ação |
|---|---|
| `0:00` | `musicgen list-genres` |
| `0:15` | `musicgen generate --genre jazz` |
| `0:45` | `ls` + inspecionar `sample.json` |
| `1:00` | Abrir notebook → seção 5 |
| `1:25` | Seção 6 (composição jazz + latin) |
| `2:00` | Seção 12 (determinismo, SHA-256) |
| `2:30` | Terminal → pytest |
| `2:50` | Encerrar + perguntas |

---

## Plano B — se algo falhar

| Problema | Solução |
|---|---|
| FluidSynth trava / demora | `Ctrl+C` → mostrar dataset pré-gerado em `/tmp/demo_dataset_backup/` (gerar antes da palestra) |
| Notebook não abre | Mostrar células como texto no terminal: `cat notebooks/musicgen_demo.ipynb \| python3 -c "import sys,json; [print(c['source']) for c in json.load(sys.stdin)['cells'] if 'genre' in ''.join(c['source'])]"` |
| pytest falha | Mostrar o JSON de resultados de benchmark: `cat benchmarks/results/*.json \| python3 -m json.tool \| head -40` |
| Erro na geração | Usar `--output-mode midi-only` (pula síntese FluidSynth): `musicgen generate --count 1 --seed 42 --out /tmp/demo_midi --genre jazz --output-mode midi-only` |

---

## Checklist final (30 min antes)

```
□ venv ativa: which python → .venv/bin/python
□ musicgen instalado: musicgen --version
□ FluidSynth: fluidsynth --version
□ sf2 presentes: ls sf/beat/ sf/melody/ sf/harmony/ sf/bassline/
□ /tmp/demo_dataset limpo: rm -rf /tmp/demo_dataset
□ /tmp/demo_dataset_backup/ gerado (plano B)
□ Notebook aberto, células 5/6/12 pré-executadas com saída visível
□ Fonte do terminal ≥ 18pt
□ pytest passando: pytest -m "not slow" -q --tb=no | tail -1
□ Benchmark rodado: benchmarks/results/ tem JSON desta máquina
```
