package com.example.task5

import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.navigation.findNavController
import com.example.task5.databinding.Fragment1Binding


class Fragment1 : Fragment(R.layout.fragment1) {
    lateinit var binding: Fragment1Binding

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        binding = Fragment1Binding.inflate(inflater, container, false)
        binding.bnToSecond.setOnClickListener {
            Log.i("fragment1", "clicked")
            it.findNavController().navigate(R.id.action_fragment1_to_fragment2)
        }
        setHasOptionsMenu(true)
        return binding.root
    }
}