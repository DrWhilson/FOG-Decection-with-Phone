package com.lipatov.fogmobile

import android.content.Intent
import android.media.MediaPlayer
import android.os.IBinder
import android.provider.Settings

class MyService: android.app.Service() {
    private lateinit var player:MediaPlayer;

    override fun onStartCommand(init:Intent, flag:Int, stratid:Int): Int {
        player =MediaPlayer.create(this,Settings.System.DEFAULT_RINGTONE_URI)
        player.setLooping(true)
        player.start()
        return START_STICKY
    }

    override fun onDestroy() {
        super.onDestroy()
        player.stop()
    }

    override fun onBind(intent: Intent?): IBinder? {
        return null
    }

}