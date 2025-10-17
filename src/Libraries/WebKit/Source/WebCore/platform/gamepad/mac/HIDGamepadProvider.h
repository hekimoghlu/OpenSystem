/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 13, 2025.
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
#pragma once

#if ENABLE(GAMEPAD) && PLATFORM(MAC)

#include "GamepadProvider.h"
#include "HIDGamepad.h"
#include "Timer.h"
#include <IOKit/hid/IOHIDManager.h>
#include <pal/spi/cocoa/IOKitSPI.h>
#include <wtf/Deque.h>
#include <wtf/HashMap.h>
#include <wtf/RetainPtr.h>
#include <wtf/WorkQueue.h>

namespace WebCore {

class GamepadProviderClient;

class HIDGamepadProvider : public GamepadProvider {
    WTF_MAKE_NONCOPYABLE(HIDGamepadProvider);
    friend class NeverDestroyed<HIDGamepadProvider>;
public:
    WEBCORE_EXPORT static HIDGamepadProvider& singleton();

    // Do nothing since this is a singleton.
    void ref() const { }
    void deref() const { }

    WEBCORE_EXPORT void startMonitoringGamepads(GamepadProviderClient&) final;
    WEBCORE_EXPORT void stopMonitoringGamepads(GamepadProviderClient&) final;
    const Vector<WeakPtr<PlatformGamepad>>& platformGamepads() final { return m_gamepadVector; }
    void playEffect(unsigned, const String&, GamepadHapticEffectType, const GamepadEffectParameters&, CompletionHandler<void(bool)>&&) final;
    void stopEffects(unsigned, const String&, CompletionHandler<void()>&&) final;

    WEBCORE_EXPORT void stopMonitoringInput();
    WEBCORE_EXPORT void startMonitoringInput();

    void deviceAdded(IOHIDDeviceRef);
    void deviceRemoved(IOHIDDeviceRef);
    void valuesChanged(IOHIDValueRef);

    void ignoreGameControllerFrameworkDevices() { m_ignoresGameControllerFrameworkDevices = true; }

    size_t numberOfConnectedGamepads() const { return m_gamepadMap.size(); };

private:
    HIDGamepadProvider();

    std::unique_ptr<HIDGamepad> removeGamepadForDevice(IOHIDDeviceRef);

    void openAndScheduleManager();
    void closeAndUnscheduleManager();

    void initialGamepadsConnectedTimerFired();
    void inputNotificationTimerFired();

    unsigned indexForNewlyConnectedDevice();

    Vector<WeakPtr<PlatformGamepad>> m_gamepadVector;
    HashMap<IOHIDDeviceRef, std::unique_ptr<HIDGamepad>> m_gamepadMap;

    RetainPtr<IOHIDManagerRef> m_manager;

    bool m_initialGamepadsConnected { false };
    bool m_ignoresGameControllerFrameworkDevices { false };

    Timer m_initialGamepadsConnectedTimer;
    Timer m_inputNotificationTimer;

#if HAVE(MULTIGAMEPADPROVIDER_SUPPORT)
    UncheckedKeyHashSet<IOHIDDeviceRef> m_gameControllerManagedGamepads;
#endif // HAVE(MULTIGAMEPADPROVIDER_SUPPORT)
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && PLATFORM(MAC)
