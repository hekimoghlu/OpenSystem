/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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

#include "EventModifierInit.h"
#include "KeypressCommand.h"
#include "UIEventWithKeyState.h"
#include <memory>
#include <wtf/Vector.h>

namespace WebCore {

class Node;
class PlatformKeyboardEvent;

class KeyboardEvent final : public UIEventWithKeyState {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(KeyboardEvent);
public:
    enum KeyLocationCode {
        DOM_KEY_LOCATION_STANDARD = 0x00,
        DOM_KEY_LOCATION_LEFT = 0x01,
        DOM_KEY_LOCATION_RIGHT = 0x02,
        DOM_KEY_LOCATION_NUMPAD = 0x03
    };

    WEBCORE_EXPORT static Ref<KeyboardEvent> create(const PlatformKeyboardEvent&, RefPtr<WindowProxy>&&);
    static Ref<KeyboardEvent> createForBindings();

    struct Init : public EventModifierInit {
        String key;
        String code;
        unsigned location;
        bool repeat;
        bool isComposing;

        // Legacy.
        AtomString keyIdentifier;
        unsigned charCode;
        unsigned keyCode;
        unsigned which;
    };

    static Ref<KeyboardEvent> create(const AtomString& type, const Init&, IsTrusted = IsTrusted::No);

    virtual ~KeyboardEvent();
    
    WEBCORE_EXPORT void initKeyboardEvent(const AtomString& type, bool canBubble, bool cancelable, RefPtr<WindowProxy>&&,
        const AtomString& keyIdentifier, unsigned location,
        bool ctrlKey, bool altKey, bool shiftKey, bool metaKey);
    
    const String& key() const { return m_key; }
    const String& code() const { return m_code; }
    const AtomString& keyIdentifier() const { return m_keyIdentifier; }
    unsigned location() const { return m_location; }
    bool repeat() const { return m_repeat; }

    const PlatformKeyboardEvent* underlyingPlatformEvent() const { return m_underlyingPlatformEvent.get(); }
    PlatformKeyboardEvent* underlyingPlatformEvent() { return m_underlyingPlatformEvent.get(); }

    WEBCORE_EXPORT int keyCode() const; // key code for keydown and keyup, character for keypress
    WEBCORE_EXPORT int charCode() const; // character code for keypress, 0 for keydown and keyup

    bool isKeyboardEvent() const final;
    unsigned which() const final;

    bool isComposing() const { return m_isComposing; }

#if PLATFORM(COCOA)
    bool handledByInputMethod() const { return m_handledByInputMethod; }
    const Vector<KeypressCommand>& keypressCommands() const { return m_keypressCommands; }
    Vector<KeypressCommand>& keypressCommands() { return m_keypressCommands; }
#endif

private:
    KeyboardEvent();
    KeyboardEvent(const PlatformKeyboardEvent&, RefPtr<WindowProxy>&&);
    KeyboardEvent(const AtomString&, const Init&, IsTrusted = IsTrusted::No);

    std::unique_ptr<PlatformKeyboardEvent> m_underlyingPlatformEvent;
    String m_key;
    String m_code;
    AtomString m_keyIdentifier;
    unsigned m_location { DOM_KEY_LOCATION_STANDARD };
    bool m_repeat { false };
    bool m_isComposing { false };
    std::optional<unsigned> m_charCode;
    std::optional<unsigned> m_keyCode;
    std::optional<unsigned> m_which;

#if PLATFORM(COCOA)
    // Commands that were sent by AppKit when interpreting the event. Doesn't include input method commands.
    bool m_handledByInputMethod { false };
    Vector<KeypressCommand> m_keypressCommands;
#endif
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_EVENT(KeyboardEvent)
