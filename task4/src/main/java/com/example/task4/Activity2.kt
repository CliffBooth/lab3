package com.example.task4

import android.content.Intent
import android.os.Bundle
import com.example.task4.databinding.Activity2Binding

class Activity2 : ActivityWithOptionsMenu() {
    private lateinit var binding: Activity2Binding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = Activity2Binding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.bnToFirst.setOnClickListener { finish() }
        binding.bnToThird.setOnClickListener {
            val intent = Intent(this, Activity3::class.java)
            startActivity(intent)
        }
        binding.btn2ToSelf.setOnClickListener {
            val intent = Intent(this, Activity2::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP)
            startActivity(intent)
        }
    }
}