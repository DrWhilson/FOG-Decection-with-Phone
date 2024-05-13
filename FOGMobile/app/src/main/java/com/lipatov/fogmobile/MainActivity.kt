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
            MainSwitch.setOnCheckedChangeListener { buttonView, isChecked ->
                if(isChecked) {
                    startService(Intent(this@MainActivity,MyService::class.java))
                    println("Standard Switch is on")
                } else {
                    stopService(Intent(this@MainActivity,MyService::class.java))
                    println("Standard Switch is off")
                }
            }
        }
    }
}