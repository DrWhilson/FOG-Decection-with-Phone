package com.lipatov.fogmobile

import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.res.Configuration
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.view.inputmethod.InputBinding
import com.lipatov.fogmobile.databinding.ActivityMainBinding
import java.util.Locale

class MainActivity : AppCompatActivity() {
    lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        loadLang()

        binding=ActivityMainBinding.inflate(layoutInflater)

        setContentView(binding.root)
        val sharedPrefs = getSharedPreferences("Settings", MODE_PRIVATE)

        binding.apply {
            switchState.setChecked(sharedPrefs.getBoolean("State", false))

            switchState.setOnCheckedChangeListener { buttonView, isChecked ->
                if (isChecked) {
                    val editor = getSharedPreferences("Settings", MODE_PRIVATE).edit()
                    editor.putBoolean("State", true)
                    editor.apply()
                    println("Click is on")
                    startService(Intent(this@MainActivity,MyService::class.java))
                    switchState.setText(getString(R.string.switch_on_text))
                } else {
                    val editor = getSharedPreferences("Settings", MODE_PRIVATE).edit()
                    editor.putBoolean("State", false)
                    editor.apply()
                    println("Click is off")
                    stopService(Intent(this@MainActivity,MyService::class.java))
                    switchState.setText(getString(R.string.switch_off_text))
                }
            }

            buttonExit.setOnClickListener {
                println("Click is exit")
                finish()
            }
            buttonConfig.setOnClickListener {
                println("Click is exit")
                launchSettings()
            }
        }
    }

    private fun launchSettings() {
        val intent = Intent(this, ConfigActivity::class.java)
        intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
        println("Try To Create Settings!")
        startActivity(intent)
    }

    private fun loadLang() {
        val SharedPref = getSharedPreferences("Settings", Context.MODE_PRIVATE)
        val Lang = SharedPref.getString("My_Lang", "")
        ChoiseLang(Lang)
    }

    private fun ChoiseLang(Lang: String?) {
        val Loc = Locale(Lang)
        Locale.setDefault(Loc)

        val config = Configuration()
        config.locale = Loc

        baseContext.resources.updateConfiguration(config, baseContext.resources.displayMetrics)

        val  editor = getSharedPreferences("Settings", Context.MODE_PRIVATE).edit()
        editor.putString("My_Lang", Lang)
        editor.apply()

    }
}