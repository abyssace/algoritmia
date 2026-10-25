# Experimentos de algoritmos y optimización

Implementaciones reproducibles de aceleración paralela, mochila voraz, búsqueda local 2-opt, recocido simulado y aproximación TSP basada en MST.

## Estructura

```text
algoritmos_optimizacion/
├── README.md
├── requirements.txt
├── src/
│   ├── algorithms.py
│   └── experiments.py
├── tests/
│   └── test_algorithms.py
├── results/
│   ├── *.csv
│   └── *.png
└── report/
    └── REPORTE.md
```

## Ejecución

Desde esta carpeta, con Python 3.10 o posterior:

```bash
python -m pip install -r requirements.txt
python -m src.experiments
python -m unittest discover -s tests -v
```

El experimento completo guarda las tablas CSV y las gráficas PNG en `results/` y actualiza el reporte en `report/REPORTE.md`. La semilla fija permite reproducir los experimentos. El cálculo de π usa procesos reales; para Windows es necesario ejecutar el módulo como se indica y no pegar el código en un intérprete interactivo.

## Entrega pública

- Repositorio público: https://github.com/abyssace/algoritmia
- Video demo (máximo 8 minutos): **pendiente de grabar y subir**.

Al publicar, sustituye estos pendientes por las URL correspondientes en `report/REPORTE.md`.
