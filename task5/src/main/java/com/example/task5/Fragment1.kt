package com.example.task5

import android.os.Bundle
import android.util.Log
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.navigation.Navigation
import androidx.navigation.findNavController
import androidx.navigation.fragment.findNavController
import com.example.task5.databinding.Fragment1Binding


class Fragment1 : Fragment(R.layout.fragment1) {
    lateinit var binding: Fragment1Binding

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        binding = Fragment1Binding.inflate(inflater, container, false)
        binding.btn1ToSecond.setOnClickListener {
            Log.i("fragment1", "clicked")
            it.findNavController().navigate(R.id.action_fragment1_to_fragment2)
        }
        return binding.root
    }
}