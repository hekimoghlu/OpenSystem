/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 23, 2023.
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

#if ENABLE(GAMEPAD) && HAVE(WIDE_GAMECONTROLLER_SUPPORT)

#include <wtf/Forward.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

OBJC_CLASS CHHapticEngine;
OBJC_CLASS GCController;

namespace WebCore {

class GameControllerHapticEffect;
struct GamepadEffectParameters;
enum class GamepadHapticEffectType : uint8_t;

class GameControllerHapticEngines final : public RefCountedAndCanMakeWeakPtr<GameControllerHapticEngines> {
    WTF_MAKE_TZONE_ALLOCATED(GameControllerHapticEngines);
public:
    static Ref<GameControllerHapticEngines> create(GCController *);
    ~GameControllerHapticEngines();

    void playEffect(GamepadHapticEffectType, const GamepadEffectParameters&, CompletionHandler<void(bool)>&&);
    void stopEffects();

    void stop(CompletionHandler<void()>&&);

    CHHapticEngine *leftHandleEngine() { return m_leftHandleEngine.get(); }
    CHHapticEngine *rightHandleEngine() { return m_rightHandleEngine.get(); }
    CHHapticEngine *leftTriggerEngine() { return m_leftTriggerEngine.get(); }
    CHHapticEngine *rightTriggerEngine() { return m_rightTriggerEngine.get(); }

private:
    explicit GameControllerHapticEngines(GCController *);

    RefPtr<GameControllerHapticEffect>& currentEffectForType(GamepadHapticEffectType);

    RetainPtr<CHHapticEngine> m_leftHandleEngine;
    RetainPtr<CHHapticEngine> m_rightHandleEngine;
    RetainPtr<CHHapticEngine> m_leftTriggerEngine;
    RetainPtr<CHHapticEngine> m_rightTriggerEngine;
    RefPtr<GameControllerHapticEffect> m_currentDualRumbleEffect;
    RefPtr<GameControllerHapticEffect> m_currentTriggerRumbleEffect;
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && HAVE(WIDE_GAMECONTROLLER_SUPPORT)
