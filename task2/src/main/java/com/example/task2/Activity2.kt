package com.example.task2

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.util.Log
import com.example.task2.databinding.Activity2Binding

class Activity2 : ActivityWithOptionsMenu() {
    lateinit var binding: Activity2Binding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = Activity2Binding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.bnToFirst.setOnClickListener { finish() }
        binding.bnToThird.setOnClickListener {
            val intent = Intent(this, Activity3::class.java)
            startActivityForResult(intent, REQUEST)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == REQUEST) {
            if (requestCode == REQUEST && resultCode == RESULT_OK) {
                finish()
            } else if (resultCode == Activity.RESULT_CANCELED) {
                Log.i("activity2", "RESULT_CANCELED!")
            }
        }
    }
}