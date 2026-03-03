# TTS Routing Service

FastAPI-сервис для синтеза речи с маршрутизацией по имени голоса. Поддерживает два бэкенда: [ElevenLabs](https://elevenlabs.io/) и [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (модели 0.6B и 1.6B). Все ответы — WAV 8kHz mono 16-bit, готовый для телефонии.

## Требования

- Python 3.10+
- CUDA GPU (для Qwen-бэкенда)
- `sox` system package (`apt install sox libsox-fmt-all`)

## Установка

```bash
pip install numpy typing_extensions
pip install -r requirements.txt
```

(`numpy` и `typing_extensions` ставятся первыми во избежание конфликтов зависимостей.)

## Конфигурация голосов

Голоса описываются в `voices.yaml`. Пример:

```yaml
voices:
  Victor:
    backend: elevenlabs
    voice_id: "YOUR_ELEVENLABS_VOICE_ID"

  Nargiz:
    backend: qwen_1_6b
    ref_audio: voices/nargiz_ref.wav
    ref_text: "Эталонный текст голоса"

  Victor_2:
    backend: qwen_0_6b
    ref_audio: voices/victor2_ref.wav
    ref_text: "Эталонный текст голоса"
```

- `backend: elevenlabs` — запросы идут через ElevenLabs API (требует `ELEVENLABS_API_KEY`)
- `backend: qwen_0_6b` / `qwen_1_6b` — локальный инференс через Qwen3-TTS; голосовой промпт извлекается из `ref_audio` при старте

Загружаются только те Qwen-модели, которые реально используются в конфиге.

## Запуск

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

При старте сервис загрузит модели и извлечёт голосовые промпты для всех Qwen-голосов.

## API

Все TTS-ответы возвращают `audio/wav` (8kHz, mono, 16-bit PCM).

### Health check

```bash
curl http://localhost:8000/health
```

### Список доступных голосов

```bash
curl http://localhost:8000/v1/voices
# {"voices": ["Victor", "Nargiz", "Victor_2"]}
```

### Синтез речи

```bash
curl -X POST http://localhost:8000/v1/tts \
  -H "Content-Type: application/json" \
  -d '{"voice": "Nargiz", "text": "Привет, как дела?"}' \
  --output output.wav
```

Ответы:
- `200 OK` — WAV-файл
- `404` — голос не найден в `voices.yaml`
- `502` — ошибка ElevenLabs API
- `500` — ошибка инференса Qwen

## Docker

```bash
docker build -t tts-service .
docker run --gpus all -p 8000:8000 \
  -v /path/to/models:/app/models \
  -e ELEVENLABS_API_KEY=sk-... \
  tts-service
```

Папка `voices/` с референсными аудио и `voices.yaml` копируются в образ при сборке. Чтобы использовать внешние файлы, монтируйте их:

```bash
docker run --gpus all -p 8000:8000 \
  -v /path/to/models:/app/models \
  -v /path/to/voices:/app/voices \
  -v /path/to/voices.yaml:/app/voices.yaml \
  -e ELEVENLABS_API_KEY=sk-... \
  tts-service
```

## Переменные окружения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `VOICES_CONFIG` | `voices.yaml` | Путь к файлу конфигурации голосов |
| `MODEL_PATH_0_6B` | `models/Qwen3-TTS-12Hz-0.6B-Base` | Путь к модели Qwen 0.6B |
| `MODEL_PATH_1_6B` | `models/Qwen3-TTS-25Hz-1.5B-Base` | Путь к модели Qwen 1.6B |
| `ELEVENLABS_API_KEY` | — | API-ключ ElevenLabs |
| `MAX_CONCURRENT_INFERENCES` | `1` | Макс. параллельных GPU-инференсов (семафор) |
| `CORS_ORIGINS` | `*` | Допустимые CORS-источники (через запятую) |
