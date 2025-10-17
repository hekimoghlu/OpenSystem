/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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
#include "MockGamepad.h"
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class MockGamepadProvider : public GamepadProvider {
    WTF_MAKE_NONCOPYABLE(MockGamepadProvider);
    friend class NeverDestroyed<MockGamepadProvider>;
public:
    WEBCORE_TESTSUPPORT_EXPORT static MockGamepadProvider& singleton();

    WEBCORE_TESTSUPPORT_EXPORT void startMonitoringGamepads(GamepadProviderClient&) final;
    WEBCORE_TESTSUPPORT_EXPORT void stopMonitoringGamepads(GamepadProviderClient&) final;
    const Vector<WeakPtr<PlatformGamepad>>& platformGamepads() final { return m_connectedGamepadVector; }
    bool isMockGamepadProvider() const final { return true; }
    void playEffect(unsigned, const String&, GamepadHapticEffectType, const GamepadEffectParameters&, CompletionHandler<void(bool)>&&) final;
    void stopEffects(unsigned, const String&, CompletionHandler<void()>&&) final;
    void clearGamepadsForTesting() final;

    void setMockGamepadDetails(unsigned index, const String& gamepadID, const String& mapping, unsigned axisCount, unsigned buttonCount, bool supportsDualRumble);
    bool setMockGamepadAxisValue(unsigned index, unsigned axisIndex, double value);
    bool setMockGamepadButtonValue(unsigned index, unsigned buttonIndex, double value);
    bool connectMockGamepad(unsigned index);
    bool disconnectMockGamepad(unsigned index);

private:
    MockGamepadProvider();

    void gamepadInputActivity();

    Vector<WeakPtr<PlatformGamepad>> m_connectedGamepadVector;
    WeakHashMap<GamepadProviderClient, WeakHashSet<PlatformGamepad>>  m_invisibleGamepadsForClient;
    Vector<std::unique_ptr<MockGamepad>> m_mockGamepadVector;

    bool m_shouldScheduleActivityCallback { true };
};

}

#endif // ENABLE(GAMEPAD)
