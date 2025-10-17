/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 8, 2023.
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

#include "ActiveDOMObject.h"
#include "GamepadHapticEffectType.h"
#include "VisibilityChangeClient.h"
#include <wtf/Forward.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class DeferredPromise;
class Document;
class Gamepad;
struct GamepadEffectParameters;

class GamepadHapticActuator : public RefCounted<GamepadHapticActuator>, public ActiveDOMObject, public VisibilityChangeClient {
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    using EffectType = GamepadHapticEffectType;
    enum class Type : uint8_t { Vibration, DualRumble };
    enum class Result : uint8_t { Complete, Preempted };

    static Ref<GamepadHapticActuator> create(Document*, Type, Gamepad&);
    ~GamepadHapticActuator();

    Type type() const { return m_type; }
    bool canPlayEffectType(EffectType) const;
    void playEffect(EffectType, GamepadEffectParameters&&, Ref<DeferredPromise>&&);
    void reset(Ref<DeferredPromise>&&);

private:
    GamepadHapticActuator(Document*, Type, Gamepad&);

    Document* document();
    const Document* document() const;

    void stopEffects(CompletionHandler<void()>&&);
    RefPtr<DeferredPromise>& promiseForEffectType(EffectType);

    // ActiveDOMObject.
    void suspend(ReasonForSuspension) final;
    void stop() final;

    // VisibilityChangeClient.
    void visibilityStateChanged() final;

    Type m_type;
    WeakPtr<Gamepad> m_gamepad;
    RefPtr<DeferredPromise> m_dualRumbleEffectPromise;
    RefPtr<DeferredPromise> m_triggerRumbleEffectPromise;
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD)
