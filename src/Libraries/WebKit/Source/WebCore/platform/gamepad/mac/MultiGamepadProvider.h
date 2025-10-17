/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 23, 2025.
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

#if ENABLE(GAMEPAD) && HAVE(MULTIGAMEPADPROVIDER_SUPPORT)

#include "GamepadProvider.h"
#include "GamepadProviderClient.h"
#include "PlatformGamepad.h"
#include <wtf/Forward.h>
#include <wtf/HashSet.h>

namespace WebCore {

class MultiGamepadProvider : public GamepadProvider, public GamepadProviderClient {
public:
    virtual ~MultiGamepadProvider() = default;

    WEBCORE_EXPORT static MultiGamepadProvider& singleton();

    void setUsesOnlyHIDGamepadProvider(bool hidProviderOnly) { m_usesOnlyHIDProvider = hidProviderOnly; }

    // GamepadProvider
    void startMonitoringGamepads(GamepadProviderClient&) final;
    void stopMonitoringGamepads(GamepadProviderClient&) final;
    const Vector<WeakPtr<PlatformGamepad>>& platformGamepads() final { return m_gamepadVector; }
    bool isMockGamepadProvider() const { return false; }
    void playEffect(unsigned gamepadIndex, const String& gamepadID, GamepadHapticEffectType, const GamepadEffectParameters&, CompletionHandler<void(bool)>&&) final;
    void stopEffects(unsigned gamepadIndex, const String& gamepadID, CompletionHandler<void()>&&) final;

    // GamepadProviderClient
    void platformGamepadConnected(PlatformGamepad&, EventMakesGamepadsVisible) final;
    void platformGamepadDisconnected(PlatformGamepad&) final;
    void platformGamepadInputActivity(EventMakesGamepadsVisible) final;

protected:
    WEBCORE_EXPORT void dispatchPlatformGamepadInputActivity();

private:
    unsigned indexForNewlyConnectedDevice();

    bool m_shouldMakeGamepadsVisible { false };
    size_t m_initialGamepadsCount { 0 };
    Vector<WeakPtr<PlatformGamepad>> m_gamepadVector;

    // We create our own Gamepad type - to wrap both HID and GameController gamepads -
    // because MultiGamepadProvider needs to manage the indexes of its own gamepads
    // no matter what the HID or GameController index is.
    class PlatformGamepadWrapper : public PlatformGamepad {
    public:
        PlatformGamepadWrapper(unsigned index, PlatformGamepad* wrapped)
            : PlatformGamepad(index)
            , m_platformGamepad(wrapped)
        {
            m_id = wrapped->id();
            m_mapping = wrapped->mapping();
            m_connectTime = wrapped->connectTime();
        }

        MonotonicTime lastUpdateTime() const final { return m_platformGamepad->lastUpdateTime(); }
        const Vector<SharedGamepadValue>& axisValues() const final { return m_platformGamepad->axisValues(); }
        const Vector<SharedGamepadValue>& buttonValues() const final { return m_platformGamepad->buttonValues(); }

        ASCIILiteral source() const final { return m_platformGamepad->source(); }

    private:
        WeakPtr<PlatformGamepad> m_platformGamepad;
    };

    WeakHashMap<PlatformGamepad, std::unique_ptr<PlatformGamepadWrapper>> m_gamepadMap;
    bool m_hidImportComplete { false };
    bool m_usesOnlyHIDProvider { false };
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && HAVE(MULTIGAMEPADPROVIDER_SUPPORT)
