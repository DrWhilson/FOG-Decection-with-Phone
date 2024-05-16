package com.lipatov.fogmobile

import android.media.RingtoneManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.widget.Button

class AlertActivity : AppCompatActivity() {

    private var ringtone = RingtoneManager.getRingtone(this, RingtoneManager.getDefaultUri(RingtoneManager.TYPE_RINGTONE))

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_alert)

        val stopButton = findViewById<Button>(R.id.StopButton)
        stopButton.setOnClickListener {
            ringtone.stop()
        }

        playRingtone()
    }

    private fun playRingtone() {
        if (ringtone != null && !ringtone.isPlaying) {
            ringtone.play()
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