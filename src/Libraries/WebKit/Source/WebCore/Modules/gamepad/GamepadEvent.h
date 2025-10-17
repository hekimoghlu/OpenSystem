/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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

#include "Event.h"
#include "Gamepad.h"
#include <wtf/RefPtr.h>

namespace WebCore {

class GamepadEvent final : public Event {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(GamepadEvent);
public:
    ~GamepadEvent() = default;

    static Ref<GamepadEvent> create(const AtomString& eventType, Gamepad& gamepad)
    {
        return adoptRef(*new GamepadEvent(eventType, gamepad));
    }

    struct Init : EventInit {
        RefPtr<Gamepad> gamepad;
    };

    static Ref<GamepadEvent> create(const AtomString& eventType, const Init& initializer, IsTrusted isTrusted = IsTrusted::No)
    {
        return adoptRef(*new GamepadEvent(eventType, initializer, isTrusted));
    }

    Gamepad* gamepad() const { return m_gamepad.get(); }

private:
    explicit GamepadEvent(const AtomString& eventType, Gamepad&);
    GamepadEvent(const AtomString& eventType, const Init&, IsTrusted);

    RefPtr<Gamepad> m_gamepad;
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD)
