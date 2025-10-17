/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 28, 2023.
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

#include "GamepadHapticEffectType.h"
#include <wtf/Forward.h>
#include <wtf/HashSet.h>
#include <wtf/WeakHashSet.h>

namespace WebCore {

class GamepadProviderClient;
class PlatformGamepad;
struct GamepadEffectParameters;

class GamepadProvider {
public:
    virtual ~GamepadProvider() = default;

    WEBCORE_EXPORT static GamepadProvider& singleton();
    WEBCORE_EXPORT static void setSharedProvider(GamepadProvider&);

    virtual void startMonitoringGamepads(GamepadProviderClient&) = 0;
    virtual void stopMonitoringGamepads(GamepadProviderClient&) = 0;
    virtual const Vector<WeakPtr<PlatformGamepad>>& platformGamepads() = 0;
    virtual bool isMockGamepadProvider() const { return false; }

    virtual void playEffect(unsigned gamepadIndex, const String& gamepadID, GamepadHapticEffectType, const GamepadEffectParameters&, CompletionHandler<void(bool)>&&) = 0;
    virtual void stopEffects(unsigned gamepadIndex, const String& gamepadID, CompletionHandler<void()>&&) = 0;

    virtual void clearGamepadsForTesting() { }

protected:
    WEBCORE_EXPORT void dispatchPlatformGamepadInputActivity();
    void setShouldMakeGamepadsVisibile() { m_shouldMakeGamepadsVisible = true; }
    WeakHashSet<GamepadProviderClient> m_clients;

private:
    bool m_shouldMakeGamepadsVisible { false };
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD)
