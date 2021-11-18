package com.example.task4

import android.content.Intent
import android.os.Bundle
import android.view.Menu
import android.view.MenuItem
import androidx.appcompat.app.AppCompatActivity
import com.example.task4.databinding.Activity3Binding

class Activity3 : AppCompatActivity() {
    private lateinit var binding: Activity3Binding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = Activity3Binding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.bnToSecond.setOnClickListener { finish() }
        binding.bnToFirst.setOnClickListener {
            val intent = Intent(this, Activity1::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP)
            startActivity(intent)
        }
    }

    override fun onCreateOptionsMenu(menu: Menu?): Boolean {
        val inflater = menuInflater
        inflater.inflate(R.menu.menu, menu)
        return true
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.about -> {
                val intent = Intent(this, ActivityAbout::class.java)
                startActivity(intent)
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }

}