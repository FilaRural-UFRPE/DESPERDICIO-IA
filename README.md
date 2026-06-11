# 🧠 SmartRU AI — Microserviço de Inteligência Artificial

> Microserviço de IA para otimização do Restaurante Universitário da UFRPE, integrado ao sistema [SmartRU](https://semdesperdicio.smartru.com.br).

---

## 📌 O Problema que Resolve

O Restaurante Universitário da UFRPE enfrenta três problemas principais:

| Problema | Impacto |
|---|---|
| Desperdício de comida | Custo financeiro e ambiental |
| Falta de previsão de demanda | Produção em excesso ou em falta |
| No-shows (agendou e não foi) | Planejamento impossível |

Este microserviço usa os dados já coletados pelo SmartRU para **prever demanda**, **detectar no-shows** e **otimizar a produção** do RU.

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────────────────┐
│                    SmartRU Backend                       │
│              (FastAPI + PostgreSQL)                      │
│         https://semdesperdicio.smartru.com.br            │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP REST
                       ▼
┌─────────────────────────────────────────────────────────┐
│               SmartRU AI Microservice                    │
│                  (FastAPI + Python)                      │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │  Previsão   │  │  Detecção    │  │   Análise de  │  │
│  │  de Demanda │  │  de No-Show  │  │   Padrões     │  │
│  │  (XGBoost)  │  │  (XGBoost)   │  │   (Regras)    │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │           Pipeline de ML (APScheduler)           │   │
│  │   Coleta → Processamento → Treino → Deploy       │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Painel do Funcionário (SmartRU)             │
│         Alertas FCM + Dashboard com previsões            │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Estrutura do Repositório

```
smartru-ai/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── demand.py          # Endpoints de previsão de demanda
│   │   ├── noshow.py          # Endpoints de detecção de no-show
│   │   ├── patterns.py        # Endpoints de análise de padrões
│   │   └── health.py          # Health check
│   ├── models/
│   │   ├── __init__.py
│   │   ├── demand_model.py    # Modelo XGBoost de demanda
│   │   ├── noshow_model.py    # Modelo XGBoost de no-show
│   │   └── pattern_model.py   # Análise de padrões (regras + stats)
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── collector.py       # Coleta dados do backend SmartRU
│   │   ├── processor.py       # Limpeza e feature engineering
│   │   ├── trainer.py         # Treino e salvamento dos modelos
│   │   └── scheduler.py       # APScheduler para treino automático
│   └── utils/
│       ├── __init__.py
│       ├── db.py              # Conexão com PostgreSQL
│       ├── logger.py          # Logger centralizado
│       └── config.py          # Variáveis de ambiente
├── data/
│   ├── raw/                   # Dados brutos extraídos do banco
│   ├── processed/             # Dados processados e features
│   └── models/                # Modelos treinados salvos (.joblib)
├── scripts/
│   ├── train_all.py           # Treina todos os modelos manualmente
│   ├── evaluate.py            # Avalia métricas dos modelos
│   └── seed_test_data.py      # Gera dados de teste
├── tests/
│   ├── test_demand.py
│   ├── test_noshow.py
│   └── test_pipeline.py
├── main.py                    # Entrypoint FastAPI
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

---

## 🤖 Modelos de IA

### 1. Previsão de Demanda
**Objetivo:** Prever quantas refeições serão necessárias nos próximos 3 a 7 dias.

**Modelo:** XGBoost Regressor (leve, eficiente sem GPU)

**Features utilizadas:**
- Dia da semana (0=segunda ... 6=domingo)
- Semana do mês
- Semana do semestre
- Turno (almoço=0, jantar=1)
- Tipo de prato (Select, Leve Sabor, Essencial)
- Média histórica dos últimos 7 dias
- Média histórica do mesmo dia da semana
- Taxa de no-show histórica

**Estratégia Cold Start:** enquanto há menos de 30 dias de dados, usa média simples dos dias disponíveis.

---

### 2. Detecção de No-Show
**Objetivo:** Prever a probabilidade de um estudante não comparecer após agendar.

**Modelo:** XGBoost Classifier com saída de probabilidade

**Features por agendamento:**
- Taxa histórica de no-show do usuário
- Dias desde o último comparecimento
- Horário do agendamento
- Turno (almoço/jantar)
- Dia da semana
- Antecedência do agendamento (agendou com quantos dias de antecedência)
- Número de cancelamentos anteriores

**Saída:** probabilidade entre 0 e 1 (ex: 0.73 = 73% de chance de não ir)

---

### 3. Análise de Padrões
**Objetivo:** Identificar dias de pico, horários mais comuns, perfis de usuários.

**Abordagem:** Estatísticas descritivas + regras simples (sem ML pesado)

**O que gera:**
- Ranking dos dias mais movimentados da semana
- Distribuição de horários de almoço e jantar
- Taxa de no-show por dia da semana
- Tipo de refeição mais popular por turno

---

## 🔌 Endpoints da API

### Saúde
```
GET /health
```

### Previsão de Demanda
```
POST /api/demand/predict
Body: { "date": "2026-06-15", "meal_type": "lunch" }
Response: { "predicted_meals": 142, "confidence": 0.87 }

GET /api/demand/forecast?days=7
Response: [{ "date": "...", "lunch": 140, "dinner": 95 }, ...]
```

### Detecção de No-Show
```
POST /api/noshow/predict
Body: { "user_cpf": "...", "schedule_date": "...", "meal_type": "lunch" }
Response: { "noshow_probability": 0.23, "risk": "low" }

GET /api/noshow/summary?date=2026-06-15
Response: { "high_risk_count": 12, "expected_attendance": 128 }
```

### Análise de Padrões
```
GET /api/patterns/weekly
Response: { "busiest_day": "segunda", "peak_hour": "12:00", ... }

GET /api/patterns/user/{cpf}
Response: { "noshow_rate": 0.15, "preferred_meal": "lunch", ... }
```

### Treino Manual
```
POST /api/train/all          # Retreina todos os modelos
POST /api/train/demand       # Retreina só demanda
POST /api/train/noshow       # Retreina só no-show
```

---

## ⚙️ Como Rodar

### Pré-requisitos
- Python 3.11+
- PostgreSQL (mesmo banco do SmartRU)
- Variáveis de ambiente configuradas

### Instalação

```bash
git clone https://github.com/seu-usuario/smartru-ai.git
cd smartru-ai
pip install -r requirements.txt
cp .env.example .env
# edite o .env com suas credenciais
uvicorn main:app --reload --port 8001
```

### Variáveis de Ambiente (.env)
```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_DB=smart_ru
POSTGRES_USER=smart_ru_user
POSTGRES_PASSWORD=sua_senha
SMARTRU_API_URL=https://semdesperdicio.smartru.com.br/api
SMARTRU_ADMIN_API_KEY=sua_chave
RETRAIN_HOUR=3
MIN_DATA_DAYS=30
```

---

## 📦 Dependências

```
fastapi==0.115.0
uvicorn==0.30.0
xgboost==2.0.3
scikit-learn==1.4.2
pandas==2.2.2
numpy==1.26.4
psycopg2-binary==2.9.9
joblib==1.4.2
APScheduler==3.10.4
python-dotenv==1.0.0
httpx==0.27.0
```

---

## 📊 Métricas de Avaliação

| Modelo | Métrica | Meta |
|---|---|---|
| Previsão de Demanda | MAE | < 10% do valor real |
| Previsão de Demanda | RMSE | < 15 refeições |
| No-Show | F1-Score | > 0.75 |
| No-Show | ROC-AUC | > 0.80 |
| Impacto Real | Redução de desperdício | > 20% |

---

## 🗺️ Roadmap

### Fase 1 — MVP (Atual)
- [x] Previsão de demanda com XGBoost
- [x] Detecção de no-show
- [x] Análise de padrões
- [x] Treino automático diário

### Fase 2 — Integração
- [ ] Alertas automáticos via FCM para funcionários
- [ ] Dashboard de previsões no painel do funcionário
- [ ] API de recomendação de cardápio

### Fase 3 — Evolução
- [ ] Recomendação personalizada por usuário
- [ ] Modelos de séries temporais (Prophet)
- [ ] Clustering de perfis de usuário (K-Means)
- [ ] Relatórios de impacto de sustentabilidade

---

## 👥 Equipe

| Nome | Papel |
|---|---|
| Tomas José Kavela | Frontend + Integração |
| Iarley | CTO / Infraestrutura |
| Daniel | DevOps / Servidor |
| Kauan Henrique Silva Carreiro | Desenvolvimento |
| Samuel Matheus Rocha de Andrade | Desenvolvimento |
| Rivan Barroso Ferreira Junior | Desenvolvimento |

---

## 🌱 Impacto Esperado

- **Redução de 20%+** no desperdício alimentar do RU
- **Planejamento antecipado** de 3 a 7 dias de produção
- **Alertas automáticos** para funcionários antes do preparo
- **Dados reais** para tomada de decisão da gestão do RU
- **Modelo replicável** para outros RUs federais do Brasil

---

> Desenvolvido por estudantes de Sistemas de Informação da UFRPE como solução sustentável para o Restaurante Universitário.
