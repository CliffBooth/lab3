package com.example.android.lifecycles.step3;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;
import android.os.SystemClock;

import java.util.Timer;
import java.util.TimerTask;

/**
 * A ViewModel used for the {@link ChronoActivity3}.
 */
public class LiveDataTimerViewModel extends ViewModel {

    private static final int ONE_SECOND = 1000;

    private MutableLiveData<Long> mElapsedTime = new MutableLiveData<>();

    private long mInitialTime;
    private final Timer timer;

    public LiveDataTimerViewModel() {
        mInitialTime = SystemClock.elapsedRealtime();
        timer = new Timer();

        // Update the elapsed time every second.
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                final long newValue = (SystemClock.elapsedRealtime() - mInitialTime) / 1000;
                // setValue() cannot be called from a background thread so post to main thread.
                mElapsedTime.postValue(newValue);
            }
        }, ONE_SECOND, ONE_SECOND);

    }

    public LiveData<Long> getElapsedTime() {
        return mElapsedTime;
    }

    @Override
    protected void onCleared() {
        super.onCleared();
        timer.cancel();
    }
}
