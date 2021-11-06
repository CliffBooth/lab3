package com.example.android.lifecycles.step5;

import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

/**
 * A ViewModel used in step 5.
 */
public class SeekBarViewModel extends ViewModel {

    public MutableLiveData<Integer> seekbarValue = new MutableLiveData<>();

}
