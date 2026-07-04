# Laboratorio 4: Planning LLM

## 📅 Información Clave
* **Fecha Límite:** Domingo 5 de Julio, 9:00 pm.
* **Formulario de Entrega:** [Google Forms](https://forms.gle/DFVAfpCM78Xd1yZXA)
* **Entorno de Ejecución:** Estricto en **Google Colab** (nada se ejecuta en local).

---

## 🎯 Objetivo y Requisitos
Diseñar algoritmos y arquitecturas de prompts avanzados (**CoT, ToT, GoT, etc.**) para resolver problemas lógicos de varios pasos dentro de una simulación virtual.

* **Modelo Obligatorio:** [`Qwen3-8B`](https://colab.research.google.com/drive/1RXxBRwhOyO96h7MN031K_-jUEZV0mvtE?usp=sharing) (No se permiten otros modelos ni versiones *finetuneadas*).
* **Configuración Determinista:** Inferencia con `temperature=0.0`.
* **Tiempo Límite:** El tiempo máximo de ejecución en Colab debe ser **menor a 2 minutos**.
* **Puntaje Mínimo:** 2 puntos.

> 🔍 **Auditoría del Leaderboard:** Se seleccionarán casos aleatorios de las entregas para replicar los resultados con Qwen3-8B ($temperature=0$). Las salidas deben ser idénticas para validar la nota.

---

## 📊 Archivos de Datos
* `Examples.json`: Contiene escenarios de ejemplo con sus secuencias de acciones óptimas.
* `Task.json`: Dataset de evaluación (solo escenarios). Tu código debe procesar este archivo y generar un JSON de salida calculando:
    * `complexity_level`
    * `target_action_sequence`

---

## 📂 Estructura del Proyecto (`planning/`)

El desarrollo se realiza dentro de la carpeta `planning/` con la siguiente distribución de archivos:

| Archivo | Descripción |
| :--- | :--- |
| `student_agent.py` | **Aquí implementas tu código.** (Reemplaza el ejemplo base). Este archivo será leído para verificar tus resultados. |
| `llm_engine.py` | Código encargado de la carga y configuración de Qwen3-8B. |
| `dev_test.py` | Script para probar tu implementación localmente en Colab y ver tu score. |
| `submit.py` | Script para generar el archivo final `submission.json`. |
| `evaluator.py` | Contiene la métrica oficial utilizada para la evaluación. |