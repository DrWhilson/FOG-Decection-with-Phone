package com.lipatov.fogmobile

import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.MediaPlayer
import android.os.Handler
import android.os.IBinder
import android.provider.Settings
import java.sql.Time

import org.tensorflow.lite.Interpreter
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.FileInputStream

class MyService: android.app.Service() {
    private lateinit var sManager: SensorManager

    private lateinit var tflite: Interpreter
    private var accelerometerData = mutableListOf<FloatArray>()
    private lateinit var handler: Handler
    private var timeStep = 0

    val sListener = object : SensorEventListener{
        override fun onSensorChanged(event: SensorEvent?) {
            event?.values?.let { values ->
                // Collect data with a timestamp
                val currentTime = timeStep * 10
                val dataEntry = floatArrayOf(currentTime.toFloat(), values[0], values[1], values[2])
                accelerometerData.add(dataEntry)
                timeStep++
            }
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { }
    }

    override fun onStartCommand(init:Intent, flag:Int, stratid:Int): Int {
        sManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val sensor = sManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        tflite = Interpreter(loadModelFile("model.tflite"))
        handler = Handler()

        // Register Sensor
        sManager.registerListener(sListener, sensor, 100000)

        // Stop collecting data after 1 second and process it
        handler.postDelayed({
//            sManager.unregisterListener(sListener)
            processSensorData()
            timeStep = 0 // Reset time step for the next data collection cycle
        }, 1000)

        return START_STICKY
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun processSensorData() {
        // Convert list to array for the model
        val data = Array(accelerometerData.size) { i -> accelerometerData[i] }
        val output = Array(1) { FloatArray(1) }  // Adjust output size based on the model's output
        tflite.run(data, output)
        // Handle output
        handleModelOutput(output[0])
    }

    private fun handleModelOutput(output: FloatArray) {
        // Implement based on what the output represents
        println("Model output: ${output.contentToString()}")
    }

    override fun onDestroy() {
        super.onDestroy()
        sManager.unregisterListener(sListener)
    }

    override fun onBind(intent: Intent?): IBinder? {
        return null
    }
}