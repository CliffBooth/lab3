package com.example.android.lifecycles.step5;


import android.os.Bundle;

import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProvider;

import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.SeekBar;

import com.example.android.codelabs.lifecycle.R;

/**
 * Shows a SeekBar that should be synced with a value in a ViewModel.
 */
public class Fragment_step5 extends Fragment {

    private SeekBar mSeekBar;

    private SeekBarViewModel mSeekBarViewModel;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container,
                             Bundle savedInstanceState) {
        Log.i("Fragment", "created");
        // Inflate the layout for this fragment
        View root = inflater.inflate(R.layout.fragment_step5, container, false);
        mSeekBar = root.findViewById(R.id.seekBar);

        //get ViewModel
        mSeekBarViewModel = new ViewModelProvider(requireActivity()).get(SeekBarViewModel.class);
        subscribeSeekBar();

        return root;
    }

    private void subscribeSeekBar() {

        // Update the ViewModel when the SeekBar is changed.
        mSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                // Set the ViewModel's value when the change comes from the user.
                if (fromUser) {
                    Log.i("Fragment", "onProgressChanged: " + progress);
                    mSeekBarViewModel.seekbarValue.setValue(progress);
                }
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
            }

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
            }
        });

        // Update the SeekBar when the ViewModel is changed.
        mSeekBarViewModel.seekbarValue.observe(requireActivity(), (value) -> {
            if (value != null) {
                mSeekBar.setProgress(value);
            }
            Log.i("Fragment", "updating value: " + value);
        });
    }
}
