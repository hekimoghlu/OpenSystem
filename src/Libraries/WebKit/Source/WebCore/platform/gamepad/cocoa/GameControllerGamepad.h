/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 1, 2022.
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

#include "GameControllerHapticEngines.h"
#include "PlatformGamepad.h"
#include <wtf/RetainPtr.h>

OBJC_CLASS GCController;
OBJC_CLASS GCControllerAxisInput;
OBJC_CLASS GCControllerButtonInput;
OBJC_CLASS GCControllerElement;

namespace WebCore {

class GameControllerHapticEngines;

class GameControllerGamepad : public PlatformGamepad {
    WTF_MAKE_NONCOPYABLE(GameControllerGamepad);
public:
    GameControllerGamepad(GCController *, unsigned index);

    const Vector<SharedGamepadValue>& axisValues() const final { return m_axisValues; }
    const Vector<SharedGamepadValue>& buttonValues() const final { return m_buttonValues; }
    void playEffect(GamepadHapticEffectType, const GamepadEffectParameters&, CompletionHandler<void(bool)>&&) final;
    void stopEffects(CompletionHandler<void()>&&) final;

    ASCIILiteral source() const final { return "GameController"_s; }

    void noLongerHasAnyClient();

private:
    void setupElements();

#if HAVE(WIDE_GAMECONTROLLER_SUPPORT)
    GameControllerHapticEngines& ensureHapticEngines();
    Ref<GameControllerHapticEngines> ensureProtectedHapticEngines() { return ensureHapticEngines(); }
#endif

    RetainPtr<GCController> m_gcController;

    Vector<SharedGamepadValue> m_axisValues;
    Vector<SharedGamepadValue> m_buttonValues;
#if HAVE(WIDE_GAMECONTROLLER_SUPPORT)
    RefPtr<GameControllerHapticEngines> m_hapticEngines;
#endif
};



} // namespace WebCore

#endif // ENABLE(GAMEPAD)
