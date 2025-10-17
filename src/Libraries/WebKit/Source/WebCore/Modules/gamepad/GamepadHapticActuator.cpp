/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 25, 2022.
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
#include "config.h"

#if ENABLE(GAMEPAD)
#include "GamepadHapticActuator.h"

#include "Document.h"
#include "EventLoop.h"
#include "Gamepad.h"
#include "GamepadEffectParameters.h"
#include "GamepadProvider.h"
#include "JSDOMPromiseDeferred.h"
#include <wtf/CompletionHandler.h>

namespace WebCore {

static bool areEffectParametersValid(GamepadHapticEffectType effectType, const GamepadEffectParameters& parameters)
{
    if (parameters.duration < 0 || parameters.startDelay < 0)
        return false;

    if (effectType == GamepadHapticEffectType::DualRumble) {
        if (parameters.weakMagnitude < 0 || parameters.strongMagnitude < 0 || parameters.weakMagnitude > 1 || parameters.strongMagnitude > 1)
            return false;
    }
    if (effectType == GamepadHapticEffectType::TriggerRumble) {
        if (parameters.leftTrigger < 0 || parameters.rightTrigger < 0 || parameters.leftTrigger > 1 || parameters.rightTrigger > 1)
            return false;
    }
    return true;
}

Ref<GamepadHapticActuator> GamepadHapticActuator::create(Document* document, Type type, Gamepad& gamepad)
{
    auto actuator = adoptRef(*new GamepadHapticActuator(document, type, gamepad));
    actuator->suspendIfNeeded();
    return actuator;
}

GamepadHapticActuator::GamepadHapticActuator(Document* document, Type type, Gamepad& gamepad)
    : ActiveDOMObject(document)
    , m_type { type }
    , m_gamepad { gamepad }
{
    if (document)
        document->registerForVisibilityStateChangedCallbacks(*this);
}

GamepadHapticActuator::~GamepadHapticActuator() = default;

bool GamepadHapticActuator::canPlayEffectType(EffectType effectType) const
{
    if (effectType == EffectType::TriggerRumble && (!document() || !document()->settings().gamepadTriggerRumbleEnabled()))
        return false;

    return m_gamepad && m_gamepad->supportedEffectTypes().contains(effectType);
}

void GamepadHapticActuator::playEffect(EffectType effectType, GamepadEffectParameters&& effectParameters, Ref<DeferredPromise>&& promise)
{
    if (!areEffectParametersValid(effectType, effectParameters)) {
        promise->reject(Exception { ExceptionCode::TypeError, "Invalid effect parameter"_s });
        return;
    }

    auto document = this->document();
    if (!document || !document->isFullyActive() || document->hidden() || !m_gamepad) {
        promise->resolve<IDLEnumeration<Result>>(Result::Preempted);
        return;
    }
    auto& currentEffectPromise = promiseForEffectType(effectType);
    if (auto playingEffectPromise = std::exchange(currentEffectPromise, nullptr)) {
        queueTaskKeepingObjectAlive(*this, TaskSource::Gamepad, [playingEffectPromise = WTFMove(playingEffectPromise)] {
            playingEffectPromise->resolve<IDLEnumeration<Result>>(Result::Preempted);
        });
    }
    if (!canPlayEffectType(effectType)) {
        promise->reject(Exception { ExceptionCode::NotSupportedError, "This gamepad doesn't support playing such effect"_s });
        return;
    }

    effectParameters.duration = std::min(effectParameters.duration, GamepadEffectParameters::maximumDuration.milliseconds());

    currentEffectPromise = WTFMove(promise);
    GamepadProvider::singleton().playEffect(m_gamepad->index(), m_gamepad->id(), effectType, effectParameters, [this, protectedThis = makePendingActivity(*this), playingEffectPromise = currentEffectPromise, effectType](bool success) mutable {
        auto& currentEffectPromise = promiseForEffectType(effectType);
        if (playingEffectPromise != currentEffectPromise)
            return; // Was already pre-empted.
        queueTaskKeepingObjectAlive(*this, TaskSource::Gamepad, [playingEffectPromise = std::exchange(currentEffectPromise, nullptr), success] {
            playingEffectPromise->resolve<IDLEnumeration<Result>>(success ? Result::Complete : Result::Preempted);
        });
    });
}

void GamepadHapticActuator::reset(Ref<DeferredPromise>&& promise)
{
    auto document = this->document();
    if (!document || !document->isFullyActive() || document->hidden() || !m_gamepad) {
        promise->resolve<IDLEnumeration<Result>>(Result::Preempted);
        return;
    }
    stopEffects([this, protectedThis = makePendingActivity(*this), promise = WTFMove(promise)]() mutable {
        queueTaskKeepingObjectAlive(*this, TaskSource::Gamepad, [promise = WTFMove(promise)] {
            promise->resolve<IDLEnumeration<Result>>(Result::Complete);
        });
    });
}

void GamepadHapticActuator::stopEffects(CompletionHandler<void()>&& completionHandler)
{
    if (!m_triggerRumbleEffectPromise && !m_dualRumbleEffectPromise)
        return completionHandler();

    auto dualRumbleEffectPromise = std::exchange(m_dualRumbleEffectPromise, nullptr);
    auto triggerRumbleEffectPromise = std::exchange(m_triggerRumbleEffectPromise, nullptr);
    queueTaskKeepingObjectAlive(*this, TaskSource::Gamepad, [dualRumbleEffectPromise = WTFMove(dualRumbleEffectPromise), triggerRumbleEffectPromise = WTFMove(triggerRumbleEffectPromise)] {
        if (dualRumbleEffectPromise)
            dualRumbleEffectPromise->resolve<IDLEnumeration<Result>>(Result::Preempted);
        if (triggerRumbleEffectPromise)
            triggerRumbleEffectPromise->resolve<IDLEnumeration<Result>>(Result::Preempted);
    });
    GamepadProvider::singleton().stopEffects(m_gamepad->index(), m_gamepad->id(), WTFMove(completionHandler));
}

Document* GamepadHapticActuator::document()
{
    return downcast<Document>(scriptExecutionContext());
}

const Document* GamepadHapticActuator::document() const
{
    return downcast<Document>(scriptExecutionContext());
}

void GamepadHapticActuator::suspend(ReasonForSuspension)
{
    stopEffects([] { });
}

void GamepadHapticActuator::stop()
{
    stopEffects([] { });
}

void GamepadHapticActuator::visibilityStateChanged()
{
    RefPtr document = this->document();
    if (!document || !document->hidden())
        return;
    stopEffects([] { });
}

RefPtr<DeferredPromise>& GamepadHapticActuator::promiseForEffectType(EffectType effectType)
{
    switch (effectType) {
    case EffectType::TriggerRumble:
        return m_triggerRumbleEffectPromise;
    case EffectType::DualRumble:
        break;
    }
    return m_dualRumbleEffectPromise;
}

} // namespace WebCore

#endif // ENABLE(GAMEPAD)
