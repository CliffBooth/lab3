package com.example.task4

import android.content.Intent
import android.os.Bundle
import com.example.task4.databinding.Activity3Binding

class Activity3 : ActivityWithOptionsMenu() {
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
}