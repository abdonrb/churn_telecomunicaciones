# 🧠 Churn Prediction en Telecomunicaciones

Este proyecto tiene como objetivo **predecir qué clientes de una empresa de telecomunicaciones tienen mayor probabilidad de darse de baja (churn)**, con un enfoque especial en **maximizar la detección de aquellos que efectivamente se van**, incluso si eso implica predecir falsamente que algunos clientes se irán cuando en realidad se quedan.

---

## 🎯 Objetivo del proyecto

Las empresas de telecomunicaciones pierden millones por año debido a la fuga de clientes. Este modelo busca **identificar proactivamente a los clientes con alta probabilidad de abandonar**, con el fin de aplicar estrategias de retención personalizadas (ofertas, llamadas, etc.).

🔎 En este proyecto **se prioriza el recall de la clase 1 (clientes que se van)**, ya que **es preferible equivocarse prediciendo que un cliente se va cuando no lo hace, que no detectar a uno que sí se va realmente**.

---

## 📊 Datos utilizados

Se utilizó un dataset público que incluye información sobre:

- Servicios contratados (Internet, líneas múltiples, soporte técnico)
- Tipo de contrato y método de pago
- Tiempo de permanencia (`tenure`)
- Cargos mensuales y totales
- Variables personales (edad, dependientes, pareja, etc.)
- Columna objetivo: `Churn` (Sí/No)

---

## 🧪 Técnicas aplicadas

- **Análisis exploratorio de datos**
- **Codificación de variables categóricas** (OneHotEncoder)
- **Ingeniería de características** para añadir indicadores como servicios adicionales contratados
- **Resampling solo en los datos de entrenamiento** para balancear las clases
- **Ajuste del umbral de probabilidad** (ej. 0.35 en lugar de 0.5) para mejorar la sensibilidad del modelo hacia los casos de baja
- **Modelado con diferentes algoritmos lineales**:
  - Regresión Logística con regularización L1 (Lasso)
  - Regresión Logística con regularización L2 (Ridge)
  - ElasticNet (combinación de L1 y L2)

---

## ✅ Resultados destacados

| Modelo           | Recall (Churn) | Precision (Churn) | F1-score (Churn) |
|------------------|----------------|-------------------|------------------|
| **LogReg L1**     | 0.94           | 0.44              | 0.60             |
| **LogReg L2**     | 0.94           | 0.45              | 0.60             |
| **ElasticNet**      | 0.92           | 0.35            | 0.51             |

📌 Se seleccionó **Regresión Logística con L1 o L2** como modelo final por su excelente **recall**, manteniendo un compromiso razonable con la precisión y sin sobreajustar.

---

## 🚀 Conclusión

Este modelo puede ser integrado en procesos reales de retención para identificar clientes en riesgo y anticiparse con estrategias comerciales. Está orientado a la **detección proactiva del churn**, **maximizando la cobertura de los casos de fuga**, lo que resulta clave en industrias con fuerte competencia como las telecomunicaciones.


