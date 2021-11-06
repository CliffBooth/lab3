package com.example.android.lifecycles.step6;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.SavedStateHandle;
import androidx.lifecycle.ViewModel;

public class SavedStateViewModel extends ViewModel {
    private static final String NAME_KEY = "name";

    //Create constructor and use the LiveData from SavedStateHandle.
    private final SavedStateHandle mState;

    public SavedStateViewModel(SavedStateHandle state) {
        mState = state;
    }

    // Expose an immutable LiveData
    LiveData<String> getName() {
        return mState.getLiveData(NAME_KEY);
    }

    void saveNewName(String newName) {
        mState.set(NAME_KEY, newName);
    }
}
