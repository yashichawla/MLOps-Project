### First time setup w Airflow if running without Docker

```bash
chmod +x setup_airflow.sh start_airflow.sh stop_airflow.sh
./setup_airflow.sh
```

#### Start Airflow

```bash
./start_airflow.sh
```

This:

- Activates venv
- Loads .airflow.env (paths and config)

Starts:

- Webserver on http://localhost:8080
- Scheduler in background
