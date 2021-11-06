package com.example.task5

import android.os.Bundle
import android.view.*
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.findNavController
import com.example.task5.databinding.Fragment3Binding

class Fragment3 : Fragment(R.layout.fragment3) {

    lateinit var binding: Fragment3Binding

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        binding = Fragment3Binding.inflate(inflater, container, false)
        val navController = findNavController()
        binding.btn3ToFirst.setOnClickListener {
            navController.navigate(R.id.action_fragment3_to_fragment1)
        }
        binding.btn3ToSecond.setOnClickListener {
            navController.navigate(R.id.action_fragment3_to_fragment2)
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