package com.lipatov.fogmobile

import android.content.Context
import android.media.Ringtone
import android.media.RingtoneManager
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.widget.Button
import android.widget.Toast
import androidx.core.content.getSystemService
import java.lang.Exception

class AlertActivity : AppCompatActivity() {
    private lateinit var ringtone: Ringtone
    private lateinit var vibrator: Vibrator

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_alert)

        println("Alert!")
        ringtone = RingtoneManager.getRingtone(this, RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION))

        vibrator = getSystemService(Context.VIBRATOR_SERVICE) as Vibrator

        val stopButton = findViewById<Button>(R.id.StopButton)
        stopButton.setOnClickListener {
            println("Alert Stop!")
            ringtone.stop()
            vibrator.cancel()
            finish()
        }

        val sharedPrefs = getSharedPreferences("Settings", MODE_PRIVATE)
        println("Sound " + sharedPrefs.getBoolean("Sound Button", true))
        if(sharedPrefs.getBoolean("Sound Button", true)){
            playRingtone()
        }

        if(sharedPrefs.getBoolean("Vibration Button", true)) {
            startVibration()
        }
    }

    private fun playRingtone() {
        try {
            println("Play!")
            ringtone.play()
        } catch (e: Exception) {
            Toast.makeText(applicationContext, this.getString(R.string.Error), Toast.LENGTH_SHORT).show()
        }
    }

    private fun startVibration() {
        try {
            if (Build.VERSION.SDK_INT >= 26) {
                println("Vibrate1!")
                vibrator.vibrate(VibrationEffect.createOneShot(2000, VibrationEffect.DEFAULT_AMPLITUDE))
            } else {
                println("Vibrate2!")
                vibrator.vibrate(2000)
            }
        } catch (e: Exception) {
            Toast.makeText(applicationContext, this.getString(R.string.Error), Toast.LENGTH_SHORT).show()
        }
    }

    override fun onPause() {
        super.onPause()
//        ringtone.stop()
    }

    override fun onResume() {
        super.onResume()
//        playRingtone()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (ringtone != null && ringtone.isPlaying) {
            ringtone.stop()
        }
    }
}