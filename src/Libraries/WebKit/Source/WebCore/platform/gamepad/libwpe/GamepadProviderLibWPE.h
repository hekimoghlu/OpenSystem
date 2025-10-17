/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 17, 2022.
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

#if ENABLE(GAMEPAD) && USE(LIBWPE)

#include "GamepadProvider.h"
#include <wtf/HashMap.h>
#include <wtf/RunLoop.h>

struct wpe_gamepad;
struct wpe_gamepad_provider;
struct wpe_view_backend;

namespace WebCore {

class GamepadLibWPE;

class GamepadProviderLibWPE final : public GamepadProvider {
    WTF_MAKE_NONCOPYABLE(GamepadProviderLibWPE);
    friend class NeverDestroyed<GamepadProviderLibWPE>;

public:
    static GamepadProviderLibWPE& singleton();

    virtual ~GamepadProviderLibWPE();

    // Do nothing since this is a singleton.
    void ref() const { }
    void deref() const { }

    void startMonitoringGamepads(GamepadProviderClient&) final;
    void stopMonitoringGamepads(GamepadProviderClient&) final;
    const Vector<WeakPtr<PlatformGamepad>>& platformGamepads() final { return m_gamepadVector; }
    void playEffect(unsigned, const String&, GamepadHapticEffectType, const GamepadEffectParameters&, CompletionHandler<void(bool)>&&) final;
    void stopEffects(unsigned, const String&, CompletionHandler<void()>&&) final;

    enum class ShouldMakeGamepadsVisible : bool { No, Yes };
    void scheduleInputNotification(GamepadLibWPE&, ShouldMakeGamepadsVisible);

    struct wpe_view_backend* inputView();

private:
    GamepadProviderLibWPE();

    void gamepadConnected(uintptr_t);
    void gamepadDisconnected(uintptr_t);
    std::unique_ptr<GamepadLibWPE> removeGamepadForId(uintptr_t);

    unsigned indexForNewlyConnectedDevice();
    void initialGamepadsConnectedTimerFired();
    void inputNotificationTimerFired();

    Vector<WeakPtr<PlatformGamepad>> m_gamepadVector;
    HashMap<uintptr_t, std::unique_ptr<GamepadLibWPE>> m_gamepadMap;
    bool m_initialGamepadsConnected { false };

    std::unique_ptr<struct wpe_gamepad_provider, void (*)(struct wpe_gamepad_provider*)> m_provider;
    struct wpe_gamepad* m_lastActiveGamepad { nullptr };

    RunLoop::Timer m_initialGamepadsConnectedTimer;
    RunLoop::Timer m_inputNotificationTimer;
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && USE(LIBWPE)
