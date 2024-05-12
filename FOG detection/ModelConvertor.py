import tensorflow as tf

print(tf.__version__)

lstm_model = tf.keras.models.load_model('lstm_model_TEST.keras', compile=False)

export_dir = "/Models/Model"
tf.saved_model.save(lstm_model, export_dir)
