/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 25, 2022.
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

#if ENABLE(GAMEPAD) && OS(LINUX)

#include "GamepadProvider.h"
#include <libmanette.h>
#include <wtf/HashMap.h>
#include <wtf/RunLoop.h>

namespace WebCore {

class ManetteGamepad;
class GamepadProviderClient;

class ManetteGamepadProvider final : public GamepadProvider {
    WTF_MAKE_NONCOPYABLE(ManetteGamepadProvider);
    friend class NeverDestroyed<ManetteGamepadProvider>;
public:
    static ManetteGamepadProvider& singleton();

    // Do nothing since this is a singleton.
    void ref() const { }
    void deref() const { }

    virtual ~ManetteGamepadProvider();

    void startMonitoringGamepads(GamepadProviderClient&) final;
    void stopMonitoringGamepads(GamepadProviderClient&) final;
    const Vector<WeakPtr<PlatformGamepad>>& platformGamepads() final { return m_gamepadVector; }
    void playEffect(unsigned, const String&, GamepadHapticEffectType, const GamepadEffectParameters&, CompletionHandler<void(bool)>&&) final;
    void stopEffects(unsigned, const String&, CompletionHandler<void()>&&) final;

    void deviceConnected(ManetteDevice*);
    void deviceDisconnected(ManetteDevice*);

    enum class ShouldMakeGamepadsVisible : bool { No, Yes };
    void gamepadHadInput(ManetteGamepad&, ShouldMakeGamepadsVisible);

private:
    ManetteGamepadProvider();

    std::unique_ptr<ManetteGamepad> removeGamepadForDevice(ManetteDevice*);

    unsigned indexForNewlyConnectedDevice();
    void initialGamepadsConnectedTimerFired();
    void inputNotificationTimerFired();

    Vector<WeakPtr<PlatformGamepad>> m_gamepadVector;
    HashMap<ManetteDevice*, std::unique_ptr<ManetteGamepad>> m_gamepadMap;
    bool m_initialGamepadsConnected { false };

    GRefPtr<ManetteMonitor> m_monitor;
    RunLoop::Timer m_initialGamepadsConnectedTimer;
    RunLoop::Timer m_inputNotificationTimer;
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && OS(LINUX)
