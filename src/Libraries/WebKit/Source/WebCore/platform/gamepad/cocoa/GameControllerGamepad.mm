/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 17, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#import "config.h"
#import "GameControllerGamepad.h"

#if ENABLE(GAMEPAD)
#import "GameControllerGamepadProvider.h"
#import "GameControllerHapticEngines.h"
#import "GamepadConstants.h"
#import <GameController/GCControllerElement.h>
#import <GameController/GameController.h>
#import <wtf/RuntimeApplicationChecks.h>
#import <wtf/text/MakeString.h>

#if PLATFORM(IOS_FAMILY)
#import <wtf/cocoa/RuntimeApplicationChecksCocoa.h>
#endif

#import "GameControllerSoftLink.h"

namespace WebCore {

GameControllerGamepad::GameControllerGamepad(GCController *controller, unsigned index)
    : PlatformGamepad(index)
    , m_gcController(controller)
{
    ASSERT(index < 4);
    controller.playerIndex = (GCControllerPlayerIndex)(GCControllerPlayerIndex1 + index);

    setupElements();
}

static void disableDefaultSystemAction(GCControllerButtonInput *button)
{
    if ([button respondsToSelector:@selector(preferredSystemGestureState)])
        button.preferredSystemGestureState = GCSystemGestureStateDisabled;
}

void GameControllerGamepad::setupElements()
{
#if PLATFORM(IOS_FAMILY)
    // rdar://103093747 - Backbone controller not recognized by Backbone app
    if (WTF::IOSApplication::isBackboneApp() && !linkedOnOrAfterSDKWithBehavior(SDKAlignedBehavior::UsesGameControllerPhysicalInputProfile))
        m_gcController.get().extendedGamepad.valueChangedHandler = ^(GCExtendedGamepad *, GCControllerElement *) { };
#endif

    auto *profile = m_gcController.get().physicalInputProfile;

    // The user can expose an already-connected game controller to a web page by expressing explicit intent.
    // Examples include pressing a button, or wiggling the joystick with intent.
    if ([profile respondsToSelector:@selector(setThumbstickUserIntentHandler:)]) {
        [profile setThumbstickUserIntentHandler:^(__kindof GCPhysicalInputProfile*, GCControllerElement*) {
            m_lastUpdateTime = MonotonicTime::now();
            GameControllerGamepadProvider::singleton().gamepadHadInput(*this, true);
        }];
    }

    auto *homeButton = profile.buttons[GCInputButtonHome];
    m_buttonValues.resize(homeButton ? numberOfStandardGamepadButtonsWithHomeButton : numberOfStandardGamepadButtonsWithoutHomeButton);

    m_id = makeString(String(m_gcController.get().vendorName), m_gcController.get().extendedGamepad ? " Extended Gamepad"_s : " Gamepad"_s);

#if HAVE(WIDE_GAMECONTROLLER_SUPPORT)
    if (auto *haptics = [m_gcController haptics]) {
        if (canLoad_GameController_GCHapticsLocalityLeftHandle() && canLoad_GameController_GCHapticsLocalityRightHandle()) {
            if ([haptics.supportedLocalities containsObject:get_GameController_GCHapticsLocalityLeftHandle()] && [haptics.supportedLocalities containsObject:get_GameController_GCHapticsLocalityRightHandle()])
                m_supportedEffectTypes.add(GamepadHapticEffectType::DualRumble);
        }
        if (canLoad_GameController_GCHapticsLocalityLeftTrigger() && canLoad_GameController_GCHapticsLocalityRightTrigger()) {
            if ([haptics.supportedLocalities containsObject:get_GameController_GCHapticsLocalityLeftTrigger()] && [haptics.supportedLocalities containsObject:get_GameController_GCHapticsLocalityRightTrigger()])
                m_supportedEffectTypes.add(GamepadHapticEffectType::TriggerRumble);
        }
    }
#endif

    if (m_gcController.get().extendedGamepad)
        m_mapping = standardGamepadMappingString();

    auto bindButton = ^(GCControllerButtonInput *button, GamepadButtonRole index) {
        m_buttonValues[(size_t)index].setValue(button.value);
        if (!button)
            return;

        button.valueChangedHandler = ^(GCControllerButtonInput *, float value, BOOL pressed) {
            // GameController framework will materialize missing values from a HID report as NaN.
            // This shouldn't happen with physical hardware, but does happen with virtual devices
            // with imperfect reports (e.g. virtual HID devices in API tests)
            // Ignoring them is preferable to surfacing NaN to javascript.
            if (std::isnan(value))
                return;
            m_buttonValues[(size_t)index].setValue(value);
            m_lastUpdateTime = MonotonicTime::now();
            GameControllerGamepadProvider::singleton().gamepadHadInput(*this, pressed);
        };
    };

    // Button Pad
    bindButton(profile.buttons[GCInputButtonA], GamepadButtonRole::RightClusterBottom);
    bindButton(profile.buttons[GCInputButtonB], GamepadButtonRole::RightClusterRight);
    bindButton(profile.buttons[GCInputButtonX], GamepadButtonRole::RightClusterLeft);
    bindButton(profile.buttons[GCInputButtonY], GamepadButtonRole::RightClusterTop);

    // Shoulders, Triggers
    bindButton(profile.buttons[GCInputLeftShoulder], GamepadButtonRole::LeftShoulderFront);
    bindButton(profile.buttons[GCInputRightShoulder], GamepadButtonRole::RightShoulderFront);
    bindButton(profile.buttons[GCInputLeftTrigger], GamepadButtonRole::LeftShoulderBack);
    bindButton(profile.buttons[GCInputRightTrigger], GamepadButtonRole::RightShoulderBack);

    // D Pad
    bindButton(profile.dpads[GCInputDirectionPad].up, GamepadButtonRole::LeftClusterTop);
    bindButton(profile.dpads[GCInputDirectionPad].down, GamepadButtonRole::LeftClusterBottom);
    bindButton(profile.dpads[GCInputDirectionPad].left, GamepadButtonRole::LeftClusterLeft);
    bindButton(profile.dpads[GCInputDirectionPad].right, GamepadButtonRole::LeftClusterRight);
    
    // Home, Select, Start
    if (homeButton) {
        bindButton(homeButton, GamepadButtonRole::CenterClusterCenter);
        disableDefaultSystemAction(homeButton);
    }
    bindButton(profile.buttons[GCInputButtonOptions], GamepadButtonRole::CenterClusterLeft);
    disableDefaultSystemAction(profile.buttons[GCInputButtonOptions]);
    bindButton(profile.buttons[GCInputButtonMenu], GamepadButtonRole::CenterClusterRight);
    disableDefaultSystemAction(profile.buttons[GCInputButtonMenu]);

    // L3, R3
    bindButton(profile.buttons[GCInputLeftThumbstickButton], GamepadButtonRole::LeftStick);
    bindButton(profile.buttons[GCInputRightThumbstickButton], GamepadButtonRole::RightStick);

    m_axisValues.resize(4);
    m_axisValues[0].setValue(profile.dpads[GCInputLeftThumbstick].xAxis.value);
    m_axisValues[1].setValue(-profile.dpads[GCInputLeftThumbstick].yAxis.value);
    m_axisValues[2].setValue(profile.dpads[GCInputRightThumbstick].xAxis.value);
    m_axisValues[3].setValue(-profile.dpads[GCInputRightThumbstick].yAxis.value);

    profile.dpads[GCInputLeftThumbstick].xAxis.valueChangedHandler = ^(GCControllerAxisInput *, float value) {
        m_axisValues[0].setValue(value);
        m_lastUpdateTime = MonotonicTime::now();
        GameControllerGamepadProvider::singleton().gamepadHadInput(*this, false);
    };
    profile.dpads[GCInputLeftThumbstick].yAxis.valueChangedHandler = ^(GCControllerAxisInput *, float value) {
        m_axisValues[1].setValue(-value);
        m_lastUpdateTime = MonotonicTime::now();
        GameControllerGamepadProvider::singleton().gamepadHadInput(*this, false);
    };
    profile.dpads[GCInputRightThumbstick].xAxis.valueChangedHandler = ^(GCControllerAxisInput *, float value) {
        m_axisValues[2].setValue(value);
        m_lastUpdateTime = MonotonicTime::now();
        GameControllerGamepadProvider::singleton().gamepadHadInput(*this, false);
    };
    profile.dpads[GCInputRightThumbstick].yAxis.valueChangedHandler = ^(GCControllerAxisInput *, float value) {
        m_axisValues[3].setValue(-value);
        m_lastUpdateTime = MonotonicTime::now();
        GameControllerGamepadProvider::singleton().gamepadHadInput(*this, false);
    };
}

#if HAVE(WIDE_GAMECONTROLLER_SUPPORT)
GameControllerHapticEngines& GameControllerGamepad::ensureHapticEngines()
{
    if (!m_hapticEngines)
        m_hapticEngines = GameControllerHapticEngines::create(m_gcController.get());
    return *m_hapticEngines;
}
#endif

void GameControllerGamepad::playEffect(GamepadHapticEffectType type, const GamepadEffectParameters& parameters, CompletionHandler<void(bool)>&& completionHandler)
{
#if HAVE(WIDE_GAMECONTROLLER_SUPPORT)
    ensureProtectedHapticEngines()->playEffect(type, parameters, WTFMove(completionHandler));
#else
    UNUSED_PARAM(type);
    UNUSED_PARAM(parameters);
    completionHandler(false);
#endif
}

void GameControllerGamepad::stopEffects(CompletionHandler<void()>&& completionHandler)
{
#if HAVE(WIDE_GAMECONTROLLER_SUPPORT)
    if (RefPtr hapticEngines = m_hapticEngines)
        hapticEngines->stopEffects();
#endif
    completionHandler();
}

void GameControllerGamepad::noLongerHasAnyClient()
{
#if HAVE(WIDE_GAMECONTROLLER_SUPPORT)
    // Stop the haptics engine if it is running.
    if (RefPtr hapticEngines = m_hapticEngines)
        hapticEngines->stop([] { });
#endif
}

} // namespace WebCore

#endif // ENABLE(GAMEPAD)
