/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 22, 2024.
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
#include "MouseEvent.h"

#include "EventNames.h"
#include "HTMLIFrameElement.h"
#include "JSDOMConvertInterface.h"
#include "JSDOMConvertNullable.h"
#include "JSEventTarget.h"
#include "JSEventTargetCustom.h"
#include "LocalFrame.h"
#include "LocalFrameView.h"
#include "PlatformMouseEvent.h"
#include <JavaScriptCore/CallFrame.h>
#include <JavaScriptCore/JSGlobalObjectInlines.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(MouseEvent);

bool isAnyClick(const AtomString& eventType)
{
    return eventType == eventNames().clickEvent || eventType == eventNames().auxclickEvent;
}

bool isAnyClick(const Event& event)
{
    return isAnyClick(event.type());
}

Ref<MouseEvent> MouseEvent::create(const AtomString& type, const MouseEventInit& initializer)
{
    return adoptRef(*new MouseEvent(EventInterfaceType::MouseEvent, type, initializer, IsTrusted::No));
}

Ref<MouseEvent> MouseEvent::create(const AtomString& eventType, RefPtr<WindowProxy>&& view, const PlatformMouseEvent& event, const Vector<Ref<MouseEvent>>& coalescedEvents, const Vector<Ref<MouseEvent>>& predictedEvents, int detail, Node* relatedTarget)
{
    auto& eventNames = WebCore::eventNames();
    bool isMouseEnterOrLeave = eventType == eventNames.mouseenterEvent || eventType == eventNames.mouseleaveEvent;
    auto isCancelable = !isMouseEnterOrLeave ? IsCancelable::Yes : IsCancelable::No;
    auto canBubble = !isMouseEnterOrLeave ? CanBubble::Yes : CanBubble::No;
    auto isComposed = !isMouseEnterOrLeave ? IsComposed::Yes : IsComposed::No;

    return MouseEvent::create(eventType, canBubble, isCancelable, isComposed, event.timestamp().approximateMonotonicTime(), WTFMove(view), detail,
        event.globalPosition(), event.position(), event.movementDelta().x(), event.movementDelta().y(),
        event.modifiers(), event.button(), event.buttons(), relatedTarget, event.force(), event.syntheticClickType(), coalescedEvents, predictedEvents);
}

Ref<MouseEvent> MouseEvent::create(const AtomString& type, CanBubble canBubble, IsCancelable isCancelable, IsComposed isComposed, MonotonicTime timestamp, RefPtr<WindowProxy>&& view, int detail,
    const IntPoint& screenLocation, const IntPoint& windowLocation, double movementX, double movementY, OptionSet<Modifier> modifiers, MouseButton button, unsigned short buttons,
    EventTarget* relatedTarget, double force, SyntheticClickType syntheticClickType, const Vector<Ref<MouseEvent>>& coalescedEvents, const Vector<Ref<MouseEvent>>& predictedEvents, IsSimulated isSimulated, IsTrusted isTrusted)
{
    return adoptRef(*new MouseEvent(EventInterfaceType::MouseEvent, type, canBubble, isCancelable, isComposed, timestamp, WTFMove(view), detail,
        screenLocation, windowLocation, movementX, movementY, modifiers, button, buttons, relatedTarget, force, syntheticClickType, coalescedEvents, predictedEvents, isSimulated, isTrusted));
}

Ref<MouseEvent> MouseEvent::create(const AtomString& eventType, CanBubble canBubble, IsCancelable isCancelable, IsComposed isComposed, MonotonicTime timestamp, RefPtr<WindowProxy>&& view, int detail,
    int screenX, int screenY, int clientX, int clientY, OptionSet<Modifier> modifiers, MouseButton button, unsigned short buttons,
    SyntheticClickType syntheticClickType, EventTarget* relatedTarget)
{
    return adoptRef(*new MouseEvent(EventInterfaceType::MouseEvent, eventType, canBubble, isCancelable, isComposed, timestamp, WTFMove(view), detail, { screenX, screenY }, { clientX, clientY }, 0, 0, modifiers, button, buttons, syntheticClickType, relatedTarget));
}


Ref<MouseEvent> MouseEvent::createForBindings()
{
    return adoptRef(*new MouseEvent(EventInterfaceType::MouseEvent));
}

MouseEvent::MouseEvent(enum EventInterfaceType eventInterface)
    : MouseRelatedEvent(eventInterface)
{
}

MouseEvent::MouseEvent(enum EventInterfaceType eventInterface, const AtomString& eventType, CanBubble canBubble, IsCancelable isCancelable, IsComposed isComposed,
    MonotonicTime timestamp, RefPtr<WindowProxy>&& view, int detail,
    const IntPoint& screenLocation, const IntPoint& windowLocation, double movementX, double movementY, OptionSet<Modifier> modifiers, MouseButton button, unsigned short buttons,
    EventTarget* relatedTarget, double force, SyntheticClickType syntheticClickType, const Vector<Ref<MouseEvent>>& coalescedEvents, const Vector<Ref<MouseEvent>>& predictedEvents, IsSimulated isSimulated, IsTrusted isTrusted)
    : MouseRelatedEvent(eventInterface, eventType, canBubble, isCancelable, isComposed, timestamp, WTFMove(view), detail, screenLocation, windowLocation, movementX, movementY, modifiers, isSimulated, isTrusted)
    , m_button(enumToUnderlyingType(button == MouseButton::None ? MouseButton::Left : button))
    , m_buttons(buttons)
    , m_syntheticClickType(button == MouseButton::None ? SyntheticClickType::NoTap : syntheticClickType)
    , m_buttonDown(button != MouseButton::None)
    , m_relatedTarget(relatedTarget)
    , m_force(force)
    , m_coalescedEvents(coalescedEvents)
    , m_predictedEvents(predictedEvents)
{
}

MouseEvent::MouseEvent(enum EventInterfaceType eventInterface, const AtomString& eventType, CanBubble canBubble, IsCancelable isCancelable, IsComposed isComposed,
    MonotonicTime timestamp, RefPtr<WindowProxy>&& view, int detail,
    const IntPoint& screenLocation, const IntPoint& clientLocation, double movementX, double movementY, OptionSet<Modifier> modifiers, MouseButton button, unsigned short buttons,
    SyntheticClickType syntheticClickType, EventTarget* relatedTarget)
    : MouseRelatedEvent(eventInterface, eventType, canBubble, isCancelable, isComposed, timestamp, WTFMove(view), detail, screenLocation, { }, movementX, movementY, modifiers, IsSimulated::No)
    , m_button(enumToUnderlyingType(button == MouseButton::None ? MouseButton::Left : button))
    , m_buttons(buttons)
    , m_syntheticClickType(button == MouseButton::None ? SyntheticClickType::NoTap : syntheticClickType)
    , m_buttonDown(button != MouseButton::None)
    , m_relatedTarget(relatedTarget)
{
    initCoordinates(clientLocation);
}

MouseEvent::MouseEvent(enum EventInterfaceType eventInterface, const AtomString& eventType, const MouseEventInit& initializer, IsTrusted isTrusted)
    : MouseRelatedEvent(eventInterface, eventType, initializer, isTrusted)
    , m_button(initializer.button == enumToUnderlyingType(MouseButton::None) ? enumToUnderlyingType(MouseButton::Left) : initializer.button)
    , m_buttons(initializer.buttons)
    , m_buttonDown(initializer.button != enumToUnderlyingType(MouseButton::None))
    , m_relatedTarget(initializer.relatedTarget)
{
    initCoordinates({ initializer.clientX, initializer.clientY });
}

MouseEvent::~MouseEvent() = default;

void MouseEvent::initMouseEvent(const AtomString& type, bool canBubble, bool cancelable, RefPtr<WindowProxy>&& view, int detail,
    int screenX, int screenY, int clientX, int clientY, bool ctrlKey, bool altKey, bool shiftKey, bool metaKey, int16_t button, EventTarget* relatedTarget)
{
    if (isBeingDispatched())
        return;

    initUIEvent(type, canBubble, cancelable, WTFMove(view), detail);

    m_screenLocation = IntPoint(screenX, screenY);
    setModifierKeys(ctrlKey, altKey, shiftKey, metaKey);
    m_button = button == enumToUnderlyingType(MouseButton::None) ? enumToUnderlyingType(MouseButton::Left) : button;
    m_syntheticClickType = SyntheticClickType::NoTap;
    m_buttonDown = button != enumToUnderlyingType(MouseButton::None);
    m_relatedTarget = relatedTarget;

    initCoordinates(IntPoint(clientX, clientY));

    setIsSimulated(false);
}

bool MouseEvent::isMouseEvent() const
{
    return true;
}

bool MouseEvent::canTriggerActivationBehavior(const Event& event)
{
    if (!isAnyClick(event))
        return false;
    auto* mouseEvent = dynamicDowncast<MouseEvent>(event);
    return !mouseEvent || mouseEvent->button() != MouseButton::Right;
}

MouseButton MouseEvent::button() const
{
    static constexpr std::array mouseButtonCases { MouseButton::None, MouseButton::PointerHasNotChanged, MouseButton::Left, MouseButton::Middle, MouseButton::Right };
    const auto isKnownButton = WTF::anyOf(mouseButtonCases, [buttonValue = this->m_button](MouseButton button) {
        return buttonValue == enumToUnderlyingType(button);
    });
    return isKnownButton ? static_cast<MouseButton>(m_button) : MouseButton::Other;
}

unsigned MouseEvent::which() const
{
    // For the DOM, the return values for left, middle and right mouse buttons are 0, 1, 2, respectively.
    // For the Netscape "which" property, the return values for left, middle and right mouse buttons are 1, 2, 3, respectively.
    // So we must add 1.
    if (!m_buttonDown)
        return 0;
    return static_cast<unsigned>(m_button + 1);
}

RefPtr<Node> MouseEvent::toElement() const
{
    // MSIE extension - "the object toward which the user is moving the mouse pointer"
    RefPtr<EventTarget> target;
    auto& eventNames = WebCore::eventNames();
    if (type() == eventNames.mouseoutEvent || type() == eventNames.mouseleaveEvent)
        target = relatedTarget();
    else
        target = this->target();
    return dynamicDowncast<Node>(WTFMove(target));
}

RefPtr<Node> MouseEvent::fromElement() const
{
    // MSIE extension - "object from which activation or the mouse pointer is exiting during the event" (huh?)
    RefPtr<EventTarget> target;
    auto& eventNames = WebCore::eventNames();
    if (type() == eventNames.mouseoutEvent || type() == eventNames.mouseleaveEvent)
        target = this->target();
    else
        target = relatedTarget();
    return dynamicDowncast<Node>(WTFMove(target));
}

} // namespace WebCore
