package com.lipatov.fogmobile

import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import com.lipatov.fogmobile.databinding.ActivityDbactivityBinding

class DBActivity : AppCompatActivity() {
    lateinit var binding: ActivityDbactivityBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_dbactivity)

        binding= ActivityDbactivityBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.apply {

            buttonDBExit.setOnClickListener {
                println("Click is exit")
                finish()
            }
        }
    }
}