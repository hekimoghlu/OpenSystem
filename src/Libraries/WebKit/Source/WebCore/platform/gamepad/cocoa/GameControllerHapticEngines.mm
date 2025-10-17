/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 11, 2022.
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
#import "config.h"

#if ENABLE(GAMEPAD) && HAVE(WIDE_GAMECONTROLLER_SUPPORT)
#import "GameControllerHapticEngines.h"

#import "GameControllerHapticEffect.h"
#import "GamepadEffectParameters.h"
#import "GamepadHapticEffectType.h"
#import "Logging.h"
#import <GameController/GameController.h>
#import <wtf/BlockPtr.h>
#import <wtf/CallbackAggregator.h>
#import <wtf/TZoneMallocInlines.h>

#import "GameControllerSoftLink.h"
#import "CoreHapticsSoftLink.h"

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(GameControllerHapticEngines);

Ref<GameControllerHapticEngines> GameControllerHapticEngines::create(GCController *gamepad)
{
    return adoptRef(*new GameControllerHapticEngines(gamepad));
}

GameControllerHapticEngines::GameControllerHapticEngines(GCController *gamepad)
    : m_leftHandleEngine([gamepad.haptics createEngineWithLocality:get_GameController_GCHapticsLocalityLeftHandle()])
    , m_rightHandleEngine([gamepad.haptics createEngineWithLocality:get_GameController_GCHapticsLocalityRightHandle()])
    , m_leftTriggerEngine([gamepad.haptics createEngineWithLocality:get_GameController_GCHapticsLocalityLeftTrigger()])
    , m_rightTriggerEngine([gamepad.haptics createEngineWithLocality:get_GameController_GCHapticsLocalityRightTrigger()])
{
}

GameControllerHapticEngines::~GameControllerHapticEngines() = default;

RefPtr<GameControllerHapticEffect>& GameControllerHapticEngines::currentEffectForType(GamepadHapticEffectType type)
{
    switch (type) {
    case GamepadHapticEffectType::DualRumble:
        return m_currentDualRumbleEffect;
    case GamepadHapticEffectType::TriggerRumble:
        return m_currentTriggerRumbleEffect;
    }
    ASSERT_NOT_REACHED();
    return m_currentDualRumbleEffect;
}

void GameControllerHapticEngines::playEffect(GamepadHapticEffectType type, const GamepadEffectParameters& parameters, CompletionHandler<void(bool)>&& completionHandler)
{
    auto& currentEffect = currentEffectForType(type);

    // Trying to create pattern players with a 0 duration will fail. However, Games on XBox seem to use such
    // requests to stop vibrating.
    if (!parameters.duration) {
        if (RefPtr effect = std::exchange(currentEffect, nullptr))
            effect->stop();
        return completionHandler(true);
    }

    auto newEffect = GameControllerHapticEffect::create(*this, type, parameters);
    if (!newEffect)
        return completionHandler(false);

    if (RefPtr effect = currentEffect)
        effect->stop();

    currentEffect = WTFMove(newEffect);
    currentEffect->start([weakThis = WeakPtr { *this }, effect = WeakPtr { *currentEffect }, type, completionHandler = WTFMove(completionHandler)](bool success) mutable {
        ASSERT(isMainThread());

        completionHandler(success);

        RefPtr protectedThis = weakThis.get();
        if (!protectedThis)
            return;

        auto& currentEffect = protectedThis->currentEffectForType(type);
        if (currentEffect.get() == effect.get())
            currentEffect = nullptr;
    });
}

void GameControllerHapticEngines::stopEffects()
{
    if (auto currentEffect = std::exchange(m_currentDualRumbleEffect, nullptr))
        currentEffect->stop();
    if (auto currentEffect = std::exchange(m_currentTriggerRumbleEffect, nullptr))
        currentEffect->stop();
}

void GameControllerHapticEngines::stop(CompletionHandler<void()>&& completionHandler)
{
    auto callbackAggregator = MainRunLoopCallbackAggregator::create(WTFMove(completionHandler));
    [m_leftHandleEngine stopWithCompletionHandler:makeBlockPtr([callbackAggregator](NSError *error) {
        if (error)
            RELEASE_LOG_ERROR(Gamepad, "GameControllerHapticEngines::stop: Failed to stop the left handle haptic engine");
    }).get()];
    [m_rightHandleEngine stopWithCompletionHandler:makeBlockPtr([callbackAggregator](NSError *error) {
        if (error)
            RELEASE_LOG_ERROR(Gamepad, "GameControllerHapticEngines::stop: Failed to stop the right handle haptic engine");
    }).get()];
    [m_leftTriggerEngine stopWithCompletionHandler:makeBlockPtr([callbackAggregator](NSError *error) {
        if (error)
            RELEASE_LOG_ERROR(Gamepad, "GameControllerHapticEngines::stop: Failed to stop the left trigger haptic engine");
    }).get()];
    [m_rightTriggerEngine stopWithCompletionHandler:makeBlockPtr([callbackAggregator](NSError *error) {
        if (error)
            RELEASE_LOG_ERROR(Gamepad, "GameControllerHapticEngines::stop: Failed to stop the right trigger haptic engine");
    }).get()];
}

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && HAVE(WIDE_GAMECONTROLLER_SUPPORT)
