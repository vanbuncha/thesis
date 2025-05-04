# Repository for thesis - 2025

## Add models folder models to:
llm/
tts/
stt/


### tts models link :
https://alphacephei.com/vosk/models

### LLM ollama setup:
`docker exec -it ollama bash`


### db
1. bash into db
`docker exec -it elderly_care_db bash`

2. connect to psql
`psql -U elderly_user -d elderly_care_db`

3. list tables 
`\dt`

4. Query
`SELECT id, room_id, sender, role, text, created_at
  FROM messages
 ORDER BY created_at DESC
 LIMIT 50;
`
