/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 28, 2024.
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

#include <WebCore/GamepadProviderClient.h>
#include <wtf/HashSet.h>
#include <wtf/RunLoop.h>
#include <wtf/Vector.h>
#include <wtf/WeakHashSet.h>

namespace WebKit {

class UIGamepad;
class WebPageProxy;
class WebProcessPool;
class GamepadData;

class UIGamepadProvider final : public WebCore::GamepadProviderClient {
public:
    static UIGamepadProvider& singleton();

    // Do nothing since this is a singleton.
    void ref() const { }
    void deref() const { }

    void processPoolStartedUsingGamepads(WebProcessPool&);
    void processPoolStoppedUsingGamepads(WebProcessPool&);

    void viewBecameActive(WebPageProxy&);
    void viewBecameInactive(WebPageProxy&);

    Vector<GamepadData> gamepadStates() const;

#if PLATFORM(COCOA)
    static void setUsesGameControllerFramework();
#endif

    Vector<std::optional<GamepadData>> snapshotGamepads();

    size_t numberOfConnectedGamepads() const { return m_gamepads.size(); }

private:
    friend NeverDestroyed<UIGamepadProvider>;
    UIGamepadProvider();
    ~UIGamepadProvider() final;

    void startMonitoringGamepads();
    void stopMonitoringGamepads();

    void platformSetDefaultGamepadProvider();
    WebPageProxy* platformWebPageProxyForGamepadInput();
    void platformStopMonitoringInput();
    void platformStartMonitoringInput();

    void platformGamepadConnected(WebCore::PlatformGamepad&, WebCore::EventMakesGamepadsVisible) final;
    void platformGamepadDisconnected(WebCore::PlatformGamepad&) final;
    void platformGamepadInputActivity(WebCore::EventMakesGamepadsVisible) final;

    void scheduleGamepadStateSync();
    void gamepadSyncTimerFired();

#if PLATFORM(VISION)
    bool isAnyGamepadConnected() const;
#endif

    WeakHashSet<WebProcessPool> m_processPoolsUsingGamepads;

    Vector<std::unique_ptr<UIGamepad>> m_gamepads;

    RunLoop::Timer m_gamepadSyncTimer;

    bool m_isMonitoringGamepads { false };
    bool m_shouldMakeGamepadsVisibleOnSync { false };
};

}

#endif // ENABLE(GAMEPAD)
