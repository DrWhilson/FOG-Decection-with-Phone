package com.lipatov.fogmobile

import android.app.Activity
import android.content.Intent
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.inputmethod.InputBinding
import com.lipatov.fogmobile.databinding.ActivityMainBinding
import com.lipatov.fogmobile.ml.ModelLstm

class MainActivity : AppCompatActivity() {
    lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding=ActivityMainBinding.inflate(layoutInflater)

        setContentView(binding.root)

        // val model = ModelLstm.newInstance(this)

//        binding.apply {
//            MainSwitch.setOnCheckedChangeListener { buttonView, isChecked ->
//                if(isChecked) {
//                    println("Standard Switch is on")
//                    startService(Intent(this@MainActivity,MyService::class.java))
//                    MainSwitch.text = getString(R.string.switch_on_text);
//                } else {
//                    println("Standard Switch is off")
//                    stopService(Intent(this@MainActivity,MyService::class.java))
//                    MainSwitch.text = getString(R.string.switch_off_text);
//                }
//            }
//        }
    }
}