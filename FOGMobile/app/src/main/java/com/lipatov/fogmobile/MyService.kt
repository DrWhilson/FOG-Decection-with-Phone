package com.lipatov.fogmobile

import android.content.Context
import android.content.Intent
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.IBinder
import android.provider.Settings
import com.lipatov.fogmobile.ml.ModelLstm
import java.sql.Time

import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.FileInputStream


class MyService: android.app.Service() {
    private lateinit var sManager: SensorManager

    //private lateinit var tflite: Interpreter
    private var accelerometerData = mutableListOf<FloatArray>()
    private lateinit var handler: Handler
    private var timeStep = 0

    // LSTM Model
    private lateinit var lstmModel: ModelLstm

    // Listen a ACCELEROMETER
    val sListener = object : SensorEventListener {
        override fun onSensorChanged(event: SensorEvent?) {
            event?.values?.let { values ->
                // Collect data with a timestamp
                val currentTime = timeStep * 10
                val dataEntry = floatArrayOf(currentTime.toFloat(), values[0], values[1], values[2])
                println("X:" + values[0] + " Y:" + values[1] + " Z:" + values[2])
                accelerometerData.add(dataEntry)
                timeStep++
            }
        }

        override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) { }
    }

    override fun onStartCommand(init:Intent, flag:Int, stratid:Int): Int {
        println("Service on!")

        // Load Model
        lstmModel = ModelLstm.newInstance(this)

        // Set ACCELEROMETER
        sManager = getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val sensor = sManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)

        // Set Handler
        handler = Handler()

        // Register Sensor
        sManager.registerListener(sListener, sensor, 100000)

        // Stop collecting data after 1 second and process it
        handler.postDelayed({
            processSensorData()
            timeStep = 0
        }, 1000)

        return START_STICKY
    }

    private fun processSensorData() {
        // Processing the data with the LSTM model
        println("Processing data...")
        // Example: lstmModel.predict(accelerometerData)
    }

    override fun onDestroy() {
        println("Service off!")
        lstmModel.close()
        sManager.unregisterListener(sListener)
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? {
        return null
    }
}