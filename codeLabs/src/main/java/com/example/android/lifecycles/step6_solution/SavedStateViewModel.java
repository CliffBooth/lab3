package com.example.android.lifecycles.step6_solution;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.SavedStateHandle;
import androidx.lifecycle.ViewModel;

public class SavedStateViewModel extends ViewModel {

    private static final String NAME_KEY = "name";

    private SavedStateHandle mState;

    public SavedStateViewModel(SavedStateHandle savedStateHandle) {
        mState = savedStateHandle;
    }

    // Expose an immutable LiveData
    LiveData<String> getName() {
        // getLiveData obtains an object that is associated with the key wrapped in a LiveData
        // so it can be observed for changes.
        return mState.getLiveData(NAME_KEY);
    }

    void saveNewName(String newName) {
        // Sets a new value for the object associated to the key. There's no need to set it
        // as a LiveData.
        mState.set(NAME_KEY, newName);
    }
}
