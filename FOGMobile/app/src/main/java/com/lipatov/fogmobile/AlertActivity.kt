package com.lipatov.fogmobile

import android.content.Context
import android.media.RingtoneManager
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.widget.Button

class AlertActivity : AppCompatActivity() {

    private var ringtone = RingtoneManager.getRingtone(this, RingtoneManager.getDefaultUri(RingtoneManager.TYPE_RINGTONE))
    private lateinit var vibrator: Vibrator

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_alert)

        vibrator = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            val vibratorManager =
                getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
            vibratorManager.defaultVibrator
        } else {
            @Suppress("DEPRECATION")
            getSystemService(VIBRATOR_SERVICE) as Vibrator
        }

        val stopButton = findViewById<Button>(R.id.StopButton)
        stopButton.setOnClickListener {
            ringtone.stop()
            vibrator.cancel()
            finish()
        }

        playRingtone()
        startVibration()
    }

    private fun playRingtone() {
        if (ringtone != null && !ringtone.isPlaying) {
            ringtone.play()
        }
    }

    private fun startVibration() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            vibrator.vibrate(VibrationEffect.createOneShot(2000, VibrationEffect.DEFAULT_AMPLITUDE))
        } else {
            vibrator.vibrate(2000)
        }
    }

    override fun onPause() {
        super.onPause()
        ringtone.stop()
    }

    override fun onResume() {
        super.onResume()
        playRingtone()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (ringtone != null && ringtone.isPlaying) {
            ringtone.stop()
        }
    }
}