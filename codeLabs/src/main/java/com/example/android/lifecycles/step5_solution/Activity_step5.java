package com.example.android.lifecycles.step5_solution;

import android.os.Bundle;

import com.example.android.codelabs.lifecycle.R;

import androidx.appcompat.app.AppCompatActivity;

/**
 * Shows two {@link Fragment_step5} fragments.
 */
public class Activity_step5 extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_step5_solution);
    }
}
