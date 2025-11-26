#!/usr/bin/env python3
# entrenar_final_completo.py - SPARK + MLflow INTEGRADOS
import os
import splitfolders
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, Callback

# ===============================================================
# CONFIGURACIÓN MLFLOW (OPCIONAL - solo si está instalado)
# ===============================================================
try:
    import mlflow
    import mlflow.tensorflow
    MLFLOW_AVAILABLE = True
    # Configurar MLflow
    mlflow.set_tracking_uri("./mlruns")  # Carpeta local
    mlflow.set_experiment("Meat-Quality-Classification")
    print(" MLflow disponible - Activando monitoreo")
except ImportError:
    MLFLOW_AVAILABLE = False
    print(" MLflow no instalado - Continuando sin monitoreo")

# ===============================================================
# CALLBACK PARA MLFLOW (si está disponible)
# ===============================================================
class MLflowCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs and MLFLOW_AVAILABLE:
            mlflow.log_metric("train_accuracy", logs.get('accuracy'), step=epoch)
            mlflow.log_metric("train_loss", logs.get('loss'), step=epoch)
            mlflow.log_metric("val_accuracy", logs.get('val_accuracy'), step=epoch) 
            mlflow.log_metric("val_loss", logs.get('val_loss'), step=epoch)

# ===============================================================
# CONFIGURACIÓN SPARK
# ===============================================================
try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
    print(" Spark disponible")
except ImportError:
    SPARK_AVAILABLE = False
    print(" Spark no disponible")

# ===============================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ===============================================================
def main():
    # ===============================================================
    # 1. INTEGRACIÓN CON SPARK (BIG DATA)
    # ===============================================================
    if SPARK_AVAILABLE:
        spark = SparkSession.builder \
            .appName("MeatQualityTraining") \
            .config("spark.executor.memory", "1g") \
            .getOrCreate()
        
        base_dir = "/home/marlon/Descargas/ProyectoCarnesV2"
        spark_processed_path = os.path.join(base_dir, "spark_processed_dataset")
        
        if os.path.exists(spark_processed_path):
            df_spark = spark.read.option("header", "true").csv(spark_processed_path)
            print(" DATASET PROCESADO POR SPARK:")
            print(f"   - Total imágenes: {df_spark.count()}")
            print("   - Distribución por clase:")
            df_spark.groupBy("class").count().show()
        else:
            print(" No se encontró análisis de Spark")
        
        spark.stop()
    else:
        print("ℹ  Saltando integración Spark")

    # ===============================================================
    # 2. INICIAR EXPERIMENTO MLFLOW
    # ===============================================================
    if MLFLOW_AVAILABLE:
        mlflow.start_run()
        # Registrar parámetros
        mlflow.log_params({
            "learning_rate": 0.0001,
            "batch_size": 32,
            "epochs": 15,
            "image_size": 128,
            "architecture": "CNN-Custom"
        })
        print(" MLflow - Experiment tracking activado")

    # ===============================================================
    # 3. ENTRENAMIENTO CNN (TU CÓDIGO ORIGINAL)
    # ===============================================================
    print(" INICIANDO ENTRENAMIENTO CNN...")
    
    base_dir = "/home/marlon/Descargas/ProyectoCarnesV2"
    data_dir = os.path.join(base_dir, "data")
    split_dir = os.path.join(base_dir, "data_split")
    
    # División del dataset
    if not os.path.exists(split_dir):
        splitfolders.ratio(data_dir, output=split_dir, seed=42, ratio=(0.7, 0.2, 0.1))
        print(" Dataset dividido")

    # Generadores de imágenes
    datagen = ImageDataGenerator(rescale=1./255)
    
    train_gen = datagen.flow_from_directory(
        os.path.join(split_dir, "train"),
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary'
    )
    
    val_gen = datagen.flow_from_directory(
        os.path.join(split_dir, "val"),
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary'
    )
    
    test_gen = datagen.flow_from_directory(
        os.path.join(split_dir, "test"),
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        shuffle=False
    )

    # Modelo CNN
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)]
    
    if MLFLOW_AVAILABLE:
        callbacks.append(MLflowCallback())
        mlflow.tensorflow.autolog()  # Registro automático

    # Entrenamiento
    history = model.fit(
        train_gen,
        epochs=15,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )

    # ===============================================================
    # 4. EVALUACIÓN Y REGISTRO EN MLFLOW
    # ===============================================================
    test_loss, test_acc = model.evaluate(test_gen)
    print(f" Precisión final: {test_acc:.4f}")

    # Métricas adicionales
    y_pred = model.predict(test_gen)
    y_pred_classes = (y_pred > 0.5).astype(int).flatten()
    y_true = test_gen.classes
    
    print("\n REPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_true, y_pred_classes, target_names=['Fresh', 'Spoiled']))

    # Registrar métricas finales en MLflow
    if MLFLOW_AVAILABLE:
        mlflow.log_metric("final_test_accuracy", test_acc)
        mlflow.log_metric("final_test_loss", test_loss)
        
        # Matriz de confusión
        cm = confusion_matrix(y_true, y_pred_classes)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fresh', 'Spoiled'], yticklabels=['Fresh', 'Spoiled'])
        plt.title("Matriz de Confusión")
        plt.xlabel("Predicción")
        plt.ylabel("Real")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

    # ===============================================================
    # 5. GUARDAR MODELO
    # ===============================================================
    model.save(os.path.join(base_dir, "modelo_meat_quality_final.h5"))
    print(" Modelo guardado como 'modelo_meat_quality_final.h5'")

    # Cerrar MLflow
    if MLFLOW_AVAILABLE:
        mlflow.end_run()
        print(" MLflow - Experimento guardado")

    return model, test_acc

# ===============================================================
# EJECUCIÓN
# ===============================================================
if __name__ == "__main__":
    print(" ENTRENAMIENTO COMPLETO: Spark + CNN + MLflow")
    print("=" * 60)
    
    model, accuracy = main()
    
    print("\n" + "=" * 60)
    print(" ENTRENAMIENTO COMPLETADO!")
    print(f" Precisión final: {accuracy:.4f}")
    
    if MLFLOW_AVAILABLE:
        print(" Para ver métricas ejecuta: mlflow ui")
    print("=" * 60)