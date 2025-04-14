from fastapi import FastAPI, UploadFile, File 
from modelo_ultis import prediccion_modelo
import shutil
import os

app = FastAPI()

@app.post("/clasificar")
async def clasificar(file: UploadFile = File(...)):
    try:
        # Guardar temporalmente la imagen
        ruta = f"temp_{file.filename}"
        with open(ruta, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Clasificación con modelo
        resultado = prediccion_modelo(ruta)

        # Eliminar archivo temporal
        os.remove(ruta)
        print("🔍 Resultado:", resultado)

        return resultado

    except Exception as e:
        return {"error": f"Error al procesar imagen: {str(e)}"}

