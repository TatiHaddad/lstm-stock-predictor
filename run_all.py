import subprocess

def run_step(description, command):
    print(f"\n {description}...")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f" Erro ao executar: {description}")
        exit(1)
    print(f" {description} finalizado com sucesso.")

if __name__ == "__main__":
    print(" Iniciando pipeline completa...")

    # 1. Coleta dos dados (gera CSV em data/raw/)
    run_step("Executando fetch_data.py (coleta de dados)", "python app/fetch_data.py")

    # 2. Treinamento do modelo (gera model_lstm.h5 e scaler)
    run_step("Executando train.py (treinamento do modelo)", "python -m model.train")

    # 3. (Opcional) Subir API localmente
    # run_step("Iniciando FastAPI", "uvicorn app.main:app --reload")

    print("\n Pipeline finalizada com sucesso!")
