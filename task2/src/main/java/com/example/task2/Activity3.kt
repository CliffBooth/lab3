package com.example.task2

import android.app.Activity
import android.os.Bundle
import com.example.task2.databinding.Activity3Binding

class Activity3 : ActivityWithOptionsMenu() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val binding = Activity3Binding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.bnToSecond.setOnClickListener { finish() }
        binding.bnToFirst.setOnClickListener {
            setResult(Activity.RESULT_OK)
            finish()
        }
    }
}