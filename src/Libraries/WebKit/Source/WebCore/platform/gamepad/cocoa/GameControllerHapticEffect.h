/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 6, 2022.
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

#import <wtf/CompletionHandler.h>
#import <wtf/RefCountedAndCanMakeWeakPtr.h>
#import <wtf/RetainPtr.h>
#import <wtf/TZoneMalloc.h>
#import <wtf/WeakPtr.h>

OBJC_CLASS CHHapticEngine;

namespace WebCore {

class GameControllerHapticEngines;
struct GamepadEffectParameters;
enum class GamepadHapticEffectType : uint8_t;

class GameControllerHapticEffect final : public RefCountedAndCanMakeWeakPtr<GameControllerHapticEffect> {
    WTF_MAKE_TZONE_ALLOCATED(GameControllerHapticEffect);
public:
    static RefPtr<GameControllerHapticEffect> create(GameControllerHapticEngines&, GamepadHapticEffectType, const GamepadEffectParameters&);
    ~GameControllerHapticEffect();

    void start(CompletionHandler<void(bool)>&&);
    void stop();

private:
    GameControllerHapticEffect(RetainPtr<CHHapticEngine>&& leftEngine, RetainPtr<CHHapticEngine>&& rightEngine, RetainPtr<id>&& leftPlayer, RetainPtr<id>&& rightPlayer);

    void ensureStarted(Function<void(bool)>&&);
    void startEngine(CHHapticEngine *, Function<void(bool)>&&);
    void registerNotification(CHHapticEngine *, Function<void(bool)>&&);

    RetainPtr<CHHapticEngine> m_leftEngine;
    RetainPtr<CHHapticEngine> m_rightEngine;
    RetainPtr<id> m_leftPlayer;
    RetainPtr<id> m_rightPlayer;
    unsigned m_engineStarted { 0 };
    unsigned m_playerFinished { 0 };
    CompletionHandler<void(bool)> m_completionHandler;
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD) && HAVE(WIDE_GAMECONTROLLER_SUPPORT)
