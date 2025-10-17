/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 7, 2024.
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

#if ENABLE(GAMEPAD)

#include "GamepadProvider.h"
#include <wtf/HashMap.h>
#include <wtf/RetainPtr.h>
#include <wtf/RunLoop.h>

OBJC_CLASS GCController;
OBJC_CLASS NSObject;

namespace WebCore {

class GameControllerGamepad;
class GamepadProviderClient;

class GameControllerGamepadProvider : public GamepadProvider {
    WTF_MAKE_NONCOPYABLE(GameControllerGamepadProvider);
    friend class NeverDestroyed<GameControllerGamepadProvider>;
public:
    WEBCORE_EXPORT static GameControllerGamepadProvider& singleton();

    // Do nothing since this is a singleton.
    void ref() const { }
    void deref() const { }

    WEBCORE_EXPORT void startMonitoringGamepads(GamepadProviderClient&) final;
    WEBCORE_EXPORT void stopMonitoringGamepads(GamepadProviderClient&) final;
    const Vector<WeakPtr<PlatformGamepad>>& platformGamepads() final { return m_gamepadVector; }
    void playEffect(unsigned gamepadIndex, const String& gamepadID, GamepadHapticEffectType, const GamepadEffectParameters&, CompletionHandler<void(bool)>&&) final;
    void stopEffects(unsigned gamepadIndex, const String& gamepadID, CompletionHandler<void()>&&) final;

    WEBCORE_EXPORT void stopMonitoringInput();
    WEBCORE_EXPORT void startMonitoringInput();

    void gamepadHadInput(GameControllerGamepad&, bool hadButtonPresses);
    void prewarmGameControllerDevicesIfNecessary();

    void makeInvisibleGamepadsVisible();

    size_t numberOfConnectedGamepads() const { return m_gamepadMap.size(); };

#if !HAVE(GCCONTROLLER_HID_DEVICE_CHECK)
    static bool willHandleVendorAndProduct(uint16_t vendorID, uint16_t productID);
#endif

private:
    GameControllerGamepadProvider();
    virtual ~GameControllerGamepadProvider();

    enum class ConnectionVisibility {
        Visible,
        Invisible,
    };

    void controllerDidConnect(GCController *, ConnectionVisibility);
    void controllerDidDisconnect(GCController *);

    unsigned indexForNewlyConnectedDevice();

    void inputNotificationTimerFired();

    HashMap<CFTypeRef, std::unique_ptr<GameControllerGamepad>> m_gamepadMap;
    Vector<WeakPtr<PlatformGamepad>> m_gamepadVector;
    WeakHashSet<PlatformGamepad> m_invisibleGamepads;

    RetainPtr<NSObject> m_connectObserver;
    RetainPtr<NSObject> m_disconnectObserver;

    RunLoop::Timer m_inputNotificationTimer;
    bool m_shouldMakeInvisibleGamepadsVisible { false };
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD)
