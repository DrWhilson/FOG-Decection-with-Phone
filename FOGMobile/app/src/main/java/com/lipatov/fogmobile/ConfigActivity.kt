package com.lipatov.fogmobile

import android.R.id.toggle
import android.content.Context
import android.content.res.Configuration
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.lipatov.fogmobile.databinding.ActivityConfigBinding
import java.util.Locale


class ConfigActivity : AppCompatActivity() {
    lateinit var binding: ActivityConfigBinding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_config)


        binding= ActivityConfigBinding.inflate(layoutInflater)
        setContentView(binding.root)
        val sharedPrefs = getSharedPreferences("Settings", MODE_PRIVATE)

        binding.apply {
            switchLang.setChecked(sharedPrefs.getBoolean("Lang Button", false))
            switchSound.setChecked(sharedPrefs.getBoolean("Sound Button", true))
            switchVibr.setChecked(sharedPrefs.getBoolean("Vibration Button", true))

            switchLang.setOnCheckedChangeListener { buttonView, isChecked ->
                if (isChecked) {
                    val editor = getSharedPreferences("Settings", MODE_PRIVATE).edit()
                    editor.putBoolean("Lang Button", true)
                    editor.apply()

                    ChoiseLang("en")
                    LangInd.setText(getString(R.string.LangInd))
                } else {
                    val editor = getSharedPreferences("Settings", MODE_PRIVATE).edit()
                    editor.putBoolean("Lang Button", false)
                    editor.apply()

                    ChoiseLang("ru")
                    LangInd.setText(getString(R.string.LangInd))
                }
            }

            switchSound.setOnCheckedChangeListener { buttonView, isChecked ->
                if (isChecked) {
                    val editor = getSharedPreferences("Settings", MODE_PRIVATE).edit()
                    editor.putBoolean("Sound Button", true)
                    editor.apply()
                } else {
                    val editor = getSharedPreferences("Settings", MODE_PRIVATE).edit()
                    editor.putBoolean("Sound Button", false)
                    editor.apply()
                }
            }

            switchVibr.setOnCheckedChangeListener { buttonView, isChecked ->
                if (isChecked) {
                    val editor = getSharedPreferences("Settings", MODE_PRIVATE).edit()
                    editor.putBoolean("Vibration Button", true)
                    editor.apply()
                } else {
                    val editor = getSharedPreferences("Settings", MODE_PRIVATE).edit()
                    editor.putBoolean("Vibration Button", false)
                    editor.apply()
                }
            }

            buttonSExit.setOnClickListener {
                println("Click is exit")
                finish()
            }
        }
    }

    private fun loadLang() {
        val SharedPref = getSharedPreferences("Settings", Context.MODE_PRIVATE)
        val Lang = SharedPref.getString("My_Lang", "")
        ChoiseLang(Lang)
    }

    private fun ChoiseLang(Lang: String?) {
        val Loc = Locale(Lang)
        println(Loc)
        Locale.setDefault(Loc)

        val config = Configuration()
        config.locale = Loc

        baseContext.resources.updateConfiguration(config, baseContext.resources.displayMetrics)

        val  editor = getSharedPreferences("Settings", Context.MODE_PRIVATE).edit()
        editor.putString("My_Lang", Lang)
        editor.apply()
        Toast.makeText(applicationContext, this.getString(R.string.RestarMess), Toast.LENGTH_SHORT).show()
    }
}