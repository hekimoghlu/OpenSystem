/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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

#include <wtf/OptionSet.h>
#include <wtf/UUID.h>
#include <wtf/WallTime.h>

namespace WebCore {

enum class EventHandling : uint8_t {
    DispatchedToDOM     = 1 << 0,
    DefaultPrevented    = 1 << 1,
    DefaultHandled      = 1 << 2,
};

enum class PlatformEventType : uint8_t {
    NoType = 0,

    // PlatformKeyboardEvent
    KeyDown,
    KeyUp,
    RawKeyDown,
    Char,

    // PlatformMouseEvent
    MouseMoved,
    MousePressed,
    MouseReleased,
    MouseForceChanged,
    MouseForceDown,
    MouseForceUp,
    MouseScroll,

    // PlatformWheelEvent
    Wheel,

#if ENABLE(TOUCH_EVENTS)
    // PlatformTouchEvent
    TouchStart,
    TouchMove,
    TouchEnd,
    TouchCancel,
    TouchForceChange,
#endif

#if ENABLE(MAC_GESTURE_EVENTS)
    // PlatformGestureEvent
    GestureStart,
    GestureChange,
    GestureEnd,
#endif
};

enum class PlatformEventModifier : uint8_t {
    AltKey      = 1 << 0,
    ControlKey  = 1 << 1,
    MetaKey     = 1 << 2,
    ShiftKey    = 1 << 3,
    CapsLockKey = 1 << 4,

    // Never used in native platforms but added for initEvent
    AltGraphKey = 1 << 5,
};


class PlatformEvent {
public:

    using Type = PlatformEventType;
    using Modifier = PlatformEventModifier;

    Type type() const { return m_type; }

    bool shiftKey() const { return m_modifiers.contains(Modifier::ShiftKey); }
    bool controlKey() const { return m_modifiers.contains(Modifier::ControlKey); }
    bool altKey() const { return m_modifiers.contains(Modifier::AltKey); }
    bool metaKey() const { return m_modifiers.contains(Modifier::MetaKey); }

    OptionSet<Modifier> modifiers() const { return m_modifiers; }

    WallTime timestamp() const { return m_timestamp; }
    std::optional<WTF::UUID> authorizationToken() const { return m_authorizationToken; };

protected:
    PlatformEvent()
        : m_type(Type::NoType)
    {
    }

    explicit PlatformEvent(Type type)
        : m_type(type)
    {
    }

    PlatformEvent(Type type, OptionSet<Modifier> modifiers, WallTime timestamp)
        : m_timestamp(timestamp)
        , m_type(type)
        , m_modifiers(modifiers)
    {
    }

    PlatformEvent(Type type, bool shiftKey, bool ctrlKey, bool altKey, bool metaKey, WallTime timestamp)
        : m_timestamp(timestamp)
        , m_type(type)
    {
        if (shiftKey)
            m_modifiers.add(Modifier::ShiftKey);
        if (ctrlKey)
            m_modifiers.add(Modifier::ControlKey);
        if (altKey)
            m_modifiers.add(Modifier::AltKey);
        if (metaKey)
            m_modifiers.add(Modifier::MetaKey);
    }

    // Explicit protected destructor so that people don't accidentally
    // delete a PlatformEvent.
    ~PlatformEvent() = default;

    WallTime m_timestamp;
    Type m_type;
    OptionSet<Modifier> m_modifiers;
    std::optional<WTF::UUID> m_authorizationToken;
};

} // namespace WebCore
