package com.lipatov.fogmobile

import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.MediaPlayer
import android.os.IBinder
import android.provider.Settings
import java.sql.Time

import org.tensorflow.lite.Interpreter
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.FileInputStream

class TFLiteModel(private val context: Context) {
    private lateinit var tflite: Interpreter

    init {
        val model = loadModelFile("model.tflite")
        tflite = Interpreter(model)
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    fun predict(input: FloatArray): FloatArray {
        val inputBuffer = arrayOf(input)
        val outputBuffer = Array(1) { FloatArray(1) }
        tflite.run(inputBuffer, outputBuffer)
        return outputBuffer[0]
    }
}


class MyService: android.app.Service() {
    private lateinit var player:MediaPlayer;
    private lateinit var sManager: SensorManager

    override fun onStartCommand(init:Intent, flag:Int, stratid:Int): Int {
        sManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val sensor = sManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val sListener = object : SensorEventListener{
            override fun onSensorChanged(event: SensorEvent?) {
                // Get Data from ACCELEROMETER
                val value = event?.values
                val sData = "X: ${value?.get(0)} Y: ${value?.get(1)} Z: ${value?.get(2)}"
            }

            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {

            }
        }
        // Register Sensor
        sManager.registerListener(sListener, sensor, SensorManager.SENSOR_DELAY_NORMAL)

//        player =MediaPlayer.create(this,Settings.System.DEFAULT_RINGTONE_URI)
//        player.setLooping(true)
//        player.start()
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
//        player.stop()
    }

    override fun onBind(intent: Intent?): IBinder? {
        return null
    }

}