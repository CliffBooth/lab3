package com.example.task5

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.navigation.findNavController
import androidx.navigation.fragment.findNavController
import com.example.task5.databinding.Fragment2Binding

class Fragment2 : Fragment(R.layout.fragment2) {
    lateinit var binding: Fragment2Binding

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        binding = Fragment2Binding.inflate(inflater, container, false)
        val navController = findNavController()
        binding.btn2ToFirst.setOnClickListener {
            navController.navigate(R.id.action_fragment2_to_fragment1)
        }
        binding.btn2ToThird.setOnClickListener {
            navController.navigate(R.id.action_fragment2_to_fragment3)
        }
        return binding.root
    }
}