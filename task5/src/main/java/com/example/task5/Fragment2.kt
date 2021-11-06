package com.example.task5

import android.os.Bundle
import android.view.*
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.example.task5.databinding.Fragment2Binding

class Fragment2 : Fragment(R.layout.fragment2) {
    lateinit var binding: Fragment2Binding

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        binding = Fragment2Binding.inflate(inflater, container, false)
        val navController = findNavController()
        binding.btn2ToFirst.setOnClickListener {
            navController.navigate(R.id.action_fragment2_to_fragment1)
        }
        binding.btn2ToThird.setOnClickListener {
            navController.navigate(R.id.action_fragment2_to_fragment3)
        }
        setHasOptionsMenu(true)
        return binding.root
    }

    override fun onCreateOptionsMenu(menu: Menu, inflater: MenuInflater) {
        super.onCreateOptionsMenu(menu, inflater)
        inflater.inflate(R.menu.menu, menu)
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.about -> {
                findNavController().navigate(R.id.activityAbout)
                true
            }
            else -> super.onOptionsItemSelected(item)
        }
    }
}