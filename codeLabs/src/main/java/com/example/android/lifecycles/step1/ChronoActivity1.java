package com.example.android.lifecycles.step1;

import android.os.Bundle;
import androidx.appcompat.app.AppCompatActivity;
import android.widget.Chronometer;

import com.example.android.codelabs.lifecycle.R;


public class ChronoActivity1 extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Chronometer chronometer = findViewById(R.id.chronometer);

        chronometer.start();
    }
}
