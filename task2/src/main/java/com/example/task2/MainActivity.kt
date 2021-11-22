package com.example.task2

import android.content.Intent
import android.os.Bundle
import com.example.task2.databinding.Activity1Binding

class MainActivity : ActivityWithOptionsMenu() {
    private lateinit var binding: Activity1Binding
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = Activity1Binding.inflate(layoutInflater)
        setContentView(binding.root)
        binding.bnToSecond.setOnClickListener {
            val intent = Intent(this, Activity2::class.java)
            startActivity(intent)
        }
    }
}