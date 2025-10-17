/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 3, 2025.
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

#include "GamepadProviderClient.h"
#include <wtf/HashSet.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class Gamepad;
class LocalDOMWindow;
class Navigator;
class WeakPtrImplWithEventTargetData;

class GamepadManager : public GamepadProviderClient {
    WTF_MAKE_NONCOPYABLE(GamepadManager);
    friend class NeverDestroyed<GamepadManager>;
public:
    static GamepadManager& singleton();

    void platformGamepadConnected(PlatformGamepad&, EventMakesGamepadsVisible) final;
    void platformGamepadDisconnected(PlatformGamepad&) final;
    void platformGamepadInputActivity(EventMakesGamepadsVisible) final;

    void registerNavigator(Navigator&);
    void unregisterNavigator(Navigator&);
    void registerDOMWindow(LocalDOMWindow&);
    void unregisterDOMWindow(LocalDOMWindow&);

#if PLATFORM(VISION)
    void updateQuarantineStatus();
#endif

private:
    GamepadManager();

    void makeGamepadVisible(PlatformGamepad&, WeakHashSet<Navigator>&, WeakHashSet<LocalDOMWindow, WeakPtrImplWithEventTargetData>&);
    void dispatchGamepadEvent(const AtomString& eventName, PlatformGamepad&);

    void maybeStartMonitoringGamepads();
    void maybeStopMonitoringGamepads();

#if PLATFORM(VISION)
    void findUnquarantinedNavigatorsAndWindows(WeakHashSet<Navigator>&, WeakHashSet<LocalDOMWindow, WeakPtrImplWithEventTargetData>&);
#endif

    bool m_isMonitoringGamepads;

    WeakHashSet<Navigator> m_navigators;
    WeakHashSet<Navigator> m_gamepadBlindNavigators;
    WeakHashSet<LocalDOMWindow, WeakPtrImplWithEventTargetData> m_domWindows;
    WeakHashSet<LocalDOMWindow, WeakPtrImplWithEventTargetData> m_gamepadBlindDOMWindows;

#if PLATFORM(VISION)
    WeakHashSet<Navigator> m_gamepadQuarantinedNavigators;
    WeakHashSet<LocalDOMWindow, WeakPtrImplWithEventTargetData> m_gamepadQuarantinedDOMWindows;
#endif
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD)
