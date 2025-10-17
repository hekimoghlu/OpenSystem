/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 2, 2025.
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
#include "KeyboardEvent.h"

#include "Document.h"
#include "Editor.h"
#include "EventHandler.h"
#include "EventNames.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "PlatformKeyboardEvent.h"
#include "WindowsKeyboardCodes.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(KeyboardEvent);

static inline const AtomString& eventTypeForKeyboardEventType(PlatformEvent::Type type)
{
    switch (type) {
    case PlatformEvent::Type::KeyUp:
        return eventNames().keyupEvent;
    case PlatformEvent::Type::RawKeyDown:
        return eventNames().keydownEvent;
    case PlatformEvent::Type::Char:
        return eventNames().keypressEvent;
    case PlatformEvent::Type::KeyDown:
        // The caller should disambiguate the combined event into RawKeyDown or Char events.
        break;
    default:
        break;
    }
    ASSERT_NOT_REACHED();
    return eventNames().keydownEvent;
}

static inline int windowsVirtualKeyCodeWithoutLocation(int keycode)
{
    switch (keycode) {
    case VK_LCONTROL:
    case VK_RCONTROL:
        return VK_CONTROL;
    case VK_LSHIFT:
    case VK_RSHIFT:
        return VK_SHIFT;
    case VK_LMENU:
    case VK_RMENU:
        return VK_MENU;
    default:
        return keycode;
    }
}

static inline KeyboardEvent::KeyLocationCode keyLocationCode(const PlatformKeyboardEvent& key)
{
    if (key.isKeypad())
        return KeyboardEvent::DOM_KEY_LOCATION_NUMPAD;

    switch (key.windowsVirtualKeyCode()) {
    case VK_LCONTROL:
    case VK_LSHIFT:
    case VK_LMENU: // Left Option/Alt
    case VK_LWIN: // Left Command/Windows key (Natural keyboard)
        return KeyboardEvent::DOM_KEY_LOCATION_LEFT;
    case VK_RCONTROL:
    case VK_RSHIFT:
    case VK_RMENU: // Right Option/Alt
    case VK_RWIN: // Right Windows key (Natural keyboard)
#if PLATFORM(COCOA)
    // FIXME: WebCore maps the right command key to VK_APPS even though the USB HID spec.,
    // <https://www.usb.org/sites/default/files/documents/hut1_12v2.pdf>, states that it
    // should map to the same key as the right Windows key (VK_RWIN).
    case VK_APPS: // Right Command
#endif
        return KeyboardEvent::DOM_KEY_LOCATION_RIGHT;
    default:
        return KeyboardEvent::DOM_KEY_LOCATION_STANDARD;
    }
}

inline KeyboardEvent::KeyboardEvent()
     : UIEventWithKeyState(EventInterfaceType::KeyboardEvent)
{
}

static bool viewIsCompositing(WindowProxy* view)
{
    if (!view)
        return false;
    auto* window = dynamicDowncast<LocalDOMWindow>(view->window());
    return window && window->frame() && window->frame()->editor().hasComposition();
}

inline KeyboardEvent::KeyboardEvent(const PlatformKeyboardEvent& key, RefPtr<WindowProxy>&& view)
    : UIEventWithKeyState(EventInterfaceType::KeyboardEvent, eventTypeForKeyboardEventType(key.type()), CanBubble::Yes, IsCancelable::Yes, IsComposed::Yes,
        key.timestamp().approximateMonotonicTime(), view.copyRef(), 0, key.modifiers(), IsTrusted::Yes)
    , m_underlyingPlatformEvent(makeUnique<PlatformKeyboardEvent>(key))
    , m_key(key.key())
    , m_code(key.code())
    , m_keyIdentifier(AtomString { key.keyIdentifier() })
    , m_location(keyLocationCode(key))
    , m_repeat(key.isAutoRepeat())
    , m_isComposing(viewIsCompositing(view.get()))
#if USE(APPKIT) || PLATFORM(IOS_FAMILY)
    , m_handledByInputMethod(key.handledByInputMethod())
#endif
#if USE(APPKIT)
    , m_keypressCommands(key.commands())
#endif
{
}

inline KeyboardEvent::KeyboardEvent(const AtomString& eventType, const Init& initializer, IsTrusted isTrusted)
    : UIEventWithKeyState(EventInterfaceType::KeyboardEvent, eventType, initializer, isTrusted)
    , m_key(initializer.key)
    , m_code(initializer.code)
    , m_keyIdentifier(initializer.keyIdentifier)
    , m_location(initializer.location)
    , m_repeat(initializer.repeat)
    , m_isComposing(initializer.isComposing)
    , m_charCode(initializer.charCode)
    , m_keyCode(initializer.keyCode)
    , m_which(initializer.which)
{
}

KeyboardEvent::~KeyboardEvent() = default;

Ref<KeyboardEvent> KeyboardEvent::create(const PlatformKeyboardEvent& platformEvent, RefPtr<WindowProxy>&& view)
{
    return adoptRef(*new KeyboardEvent(platformEvent, WTFMove(view)));
}

Ref<KeyboardEvent> KeyboardEvent::createForBindings()
{
    return adoptRef(*new KeyboardEvent);
}

Ref<KeyboardEvent> KeyboardEvent::create(const AtomString& type, const Init& initializer, IsTrusted isTrusted)
{
    return adoptRef(*new KeyboardEvent(type, initializer, isTrusted));
}

void KeyboardEvent::initKeyboardEvent(const AtomString& type, bool canBubble, bool cancelable, RefPtr<WindowProxy>&& view,
    const AtomString& keyIdentifier, unsigned location, bool ctrlKey, bool altKey, bool shiftKey, bool metaKey)
{
    if (isBeingDispatched())
        return;

    initUIEvent(type, canBubble, cancelable, WTFMove(view), 0);

    m_keyIdentifier = keyIdentifier;
    m_location = location;

    setModifierKeys(ctrlKey, altKey, shiftKey, metaKey);

    m_charCode = std::nullopt;
    m_isComposing = false;
    m_keyCode = std::nullopt;
    m_repeat = false;
    m_underlyingPlatformEvent = nullptr;
    m_which = std::nullopt;
    m_code = { };
    m_key = { };

#if PLATFORM(COCOA)
    m_handledByInputMethod = false;
    m_keypressCommands = { };
#endif
}

int KeyboardEvent::keyCode() const
{
    if (m_keyCode)
        return m_keyCode.value();

    // IE: virtual key code for keyup/keydown, character code for keypress
    // Firefox: virtual key code for keyup/keydown, zero for keypress
    // We match IE.
    if (!m_underlyingPlatformEvent)
        return 0;
    if (type() == eventNames().keydownEvent || type() == eventNames().keyupEvent)
        return windowsVirtualKeyCodeWithoutLocation(m_underlyingPlatformEvent->windowsVirtualKeyCode());

    return charCode();
}

int KeyboardEvent::charCode() const
{
    if (m_charCode)
        return m_charCode.value();

    // IE: not supported
    // Firefox: 0 for keydown/keyup events, character code for keypress
    // We match Firefox, unless in backward compatibility mode, where we always return the character code.
    bool backwardCompatibilityMode = false;
    RefPtr window = dynamicDowncast<LocalDOMWindow>(view() ? view()->window() : nullptr);
    if (window && window->frame())
        backwardCompatibilityMode = window->frame()->eventHandler().needsKeyboardEventDisambiguationQuirks();

    if (!m_underlyingPlatformEvent || (type() != eventNames().keypressEvent && !backwardCompatibilityMode))
        return 0;
    return m_underlyingPlatformEvent->text().characterStartingAt(0);
}

bool KeyboardEvent::isKeyboardEvent() const
{
    return true;
}

unsigned KeyboardEvent::which() const
{
    // Netscape's "which" returns a virtual key code for keydown and keyup, and a character code for keypress.
    // That's exactly what IE's "keyCode" returns. So they are the same for keyboard events.
    if (m_which)
        return m_which.value();
    return static_cast<unsigned>(keyCode());
}

} // namespace WebCore
