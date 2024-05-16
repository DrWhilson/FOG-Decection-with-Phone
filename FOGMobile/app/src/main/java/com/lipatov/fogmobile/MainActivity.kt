package com.lipatov.fogmobile

import android.app.Activity
import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.inputmethod.InputBinding
import com.lipatov.fogmobile.databinding.ActivityMainBinding

class MainActivity : AppCompatActivity() {
    lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding=ActivityMainBinding.inflate(layoutInflater)

        setContentView(binding.root)

        binding.apply {
            button0N.setOnClickListener {
                println("Click is on")
                startService(Intent(this@MainActivity,MyService::class.java))
            }
            buttonOff.setOnClickListener {
                println("Click is off")
                stopService(Intent(this@MainActivity,MyService::class.java))
            }
        }
    }
}