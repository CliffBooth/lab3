package com.example.task5

import android.content.pm.ActivityInfo
import androidx.lifecycle.Lifecycle
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.launchActivity
import androidx.test.espresso.Espresso
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.action.ViewActions
import androidx.test.espresso.assertion.ViewAssertions
import androidx.test.espresso.assertion.ViewAssertions.matches
import androidx.test.espresso.matcher.ViewMatchers
import androidx.test.espresso.matcher.ViewMatchers.isDisplayed
import androidx.test.espresso.matcher.ViewMatchers.withId
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class BackStackTest {
    lateinit var scenario: ActivityScenario<MainActivity>

    @Before
    fun setUp() {
        scenario = launchActivity()
    }

    private fun checkBackStackSize(ExpectedSize: Int) {
        var counter = 0
        while (scenario.state != Lifecycle.State.DESTROYED) {
            counter++
            Espresso.pressBackUnconditionally()
        }
        Assert.assertEquals(counter, ExpectedSize)
    }


    @Test
    fun stackFirst() {
        onView(withId(R.id.fragment1))
            .check(ViewAssertions.matches(ViewMatchers.isDisplayed()))
        checkBackStackSize(1)

    }

    @Test
    fun stackFirstToSecond() {
        onView(withId(R.id.bnToSecond)).perform(ViewActions.click())
        checkBackStackSize(2)
    }

    @Test
    fun stackSecondToFirst() {
        onView(withId(R.id.bnToSecond)).perform(ViewActions.click())
        onView(withId(R.id.bnToFirst)).perform(ViewActions.click())
        checkBackStackSize(1)
    }

    @Test
    fun stackThird() {
        onView(withId(R.id.bnToSecond)).perform(ViewActions.click())
        onView(withId(R.id.bnToThird)).perform(ViewActions.click())
        checkBackStackSize(3)
    }

    @Test
    fun stackThirdToSecond() {
        onView(withId(R.id.bnToSecond)).perform(ViewActions.click())
        onView(withId(R.id.bnToThird)).perform(ViewActions.click())
        onView(withId(R.id.bnToSecond)).perform(ViewActions.click())
        checkBackStackSize(2)
    }

    @Test
    fun stackThirdToFirst() {
        onView(withId(R.id.bnToSecond)).perform(ViewActions.click())
        onView(withId(R.id.bnToThird)).perform(ViewActions.click())
        onView(withId(R.id.bnToFirst)).perform(ViewActions.click())
        checkBackStackSize(1)
    }

    @Test
    fun thirdAbout() {
        onView(withId(R.id.bnToSecond)).perform(ViewActions.click())
        onView(withId(R.id.bnToThird)).perform(ViewActions.click())
        openAbout()
        checkBackStackSize(4)
    }

    @Test
    fun secondAbout() {
        onView(withId(R.id.bnToSecond)).perform(ViewActions.click())
        openAbout()
        checkBackStackSize(3)
    }

    @Test
    fun firstAbout() {
        openAbout()
        checkBackStackSize(2)
    }

    @Test
    fun manyActions() {
        onView(withId(R.id.bnToSecond)).perform(ViewActions.click())
        onView(withId(R.id.bnToFirst)).perform(ViewActions.click())
        onView(withId(R.id.bnToSecond)).perform(ViewActions.click())
        onView(withId(R.id.bnToThird)).perform(ViewActions.click())
        onView(withId(R.id.bnToSecond)).perform(ViewActions.click())
        onView(withId(R.id.bnToFirst)).perform(ViewActions.click())
        onView(withId(R.id.bnToSecond)).perform(ViewActions.click())
        onView(withId(R.id.bnToThird)).perform(ViewActions.click())
        onView(withId(R.id.bnToFirst)).perform(ViewActions.click())
        checkBackStackSize(1)
    }

    @Test
    fun rotationSecond() {
        scenario.onActivity { activity ->
            activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }
        onView(withId(R.id.bnToSecond)).perform(ViewActions.click())
        scenario.onActivity { activity ->
            activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }
        onView(withId(R.id.fragment2)).check(matches(isDisplayed()))
        checkBackStackSize(2)
    }

    @Test
    fun rotationThird() {
        scenario.onActivity { activity ->
            activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }
        onView(withId(R.id.bnToSecond)).perform(ViewActions.click())
        onView(withId(R.id.bnToThird)).perform(ViewActions.click())
        scenario.onActivity { activity ->
            activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }
        onView(withId(R.id.fragment3)).check(matches(isDisplayed()))
        checkBackStackSize(3)
    }

    @Test
    fun rotationAbout() {
        scenario.onActivity { activity ->
            activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }
        onView(withId(R.id.bnToSecond)).perform(ViewActions.click())
        openAbout()
        scenario.onActivity { activity ->
            activity.requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }
        onView(withId(R.id.activity_about)).check(matches(isDisplayed()))
        checkBackStackSize(3)
    }

}