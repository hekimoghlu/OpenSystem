/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 27, 2023.
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

#include "EventTarget.h"
#include "MouseEventInit.h"
#include "MouseRelatedEvent.h"
#include "PlatformMouseEvent.h"

#if ENABLE(TOUCH_EVENTS) && PLATFORM(IOS_FAMILY)
#include "PlatformTouchEventIOS.h"
#endif

namespace JSC {
class CallFrame;
class JSValue;
}

namespace WebCore {

class Node;
class PlatformMouseEvent;

enum class SyntheticClickType : uint8_t;

bool isAnyClick(const AtomString& eventType);
bool isAnyClick(const Event&);

class MouseEvent : public MouseRelatedEvent {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(MouseEvent);
public:
    WEBCORE_EXPORT static Ref<MouseEvent> create(const AtomString& type, CanBubble, IsCancelable, IsComposed, MonotonicTime timestamp, RefPtr<WindowProxy>&&, int detail,
        const IntPoint& screenLocation, const IntPoint& windowLocation, double movementX, double movementY, OptionSet<Modifier>, MouseButton, unsigned short buttons,
        EventTarget* relatedTarget, double force, SyntheticClickType, const Vector<Ref<MouseEvent>>& coalescedEvents, const Vector<Ref<MouseEvent>>& predictedEvents, IsSimulated = IsSimulated::No, IsTrusted = IsTrusted::Yes);

    WEBCORE_EXPORT static Ref<MouseEvent> create(const AtomString& eventType, RefPtr<WindowProxy>&&, const PlatformMouseEvent&, const Vector<Ref<MouseEvent>>& coalescedEvents, const Vector<Ref<MouseEvent>>& predictedEvents, int detail, Node* relatedTarget);

    static Ref<MouseEvent> create(const AtomString& eventType, CanBubble, IsCancelable, IsComposed, MonotonicTime timestamp, RefPtr<WindowProxy>&&, int detail,
        int screenX, int screenY, int clientX, int clientY, OptionSet<Modifier>, MouseButton, unsigned short buttons,
        SyntheticClickType, EventTarget* relatedTarget);

    static Ref<MouseEvent> createForBindings();

    static Ref<MouseEvent> create(const AtomString& eventType, const MouseEventInit&);

#if ENABLE(TOUCH_EVENTS) && PLATFORM(IOS_FAMILY)
    static Ref<MouseEvent> create(const PlatformTouchEvent&, unsigned touchIndex, Ref<WindowProxy>&&, IsCancelable = IsCancelable::Yes);
#endif

    virtual ~MouseEvent();

    WEBCORE_EXPORT void initMouseEvent(const AtomString& type, bool canBubble, bool cancelable, RefPtr<WindowProxy>&&,
        int detail, int screenX, int screenY, int clientX, int clientY, bool ctrlKey, bool altKey, bool shiftKey, bool metaKey,
        int16_t button, EventTarget* relatedTarget);

    MouseButton button() const;
    int16_t buttonAsShort() const { return m_button; }
    unsigned short buttons() const { return m_buttons; }
    SyntheticClickType syntheticClickType() const { return m_syntheticClickType; }
    bool buttonDown() const { return m_buttonDown; }
    EventTarget* relatedTarget() const final { return m_relatedTarget.get(); }
    double force() const { return m_force; }
    void setForce(double force) { m_force = force; }

    WEBCORE_EXPORT virtual RefPtr<Node> toElement() const;
    WEBCORE_EXPORT virtual RefPtr<Node> fromElement() const;

    static bool canTriggerActivationBehavior(const Event&);

    unsigned which() const final;

    Vector<Ref<MouseEvent>> coalescedEvents() const { return m_coalescedEvents; }

    Vector<Ref<MouseEvent>> predictedEvents() const { return m_predictedEvents; }

protected:
    MouseEvent(enum EventInterfaceType, const AtomString& type, CanBubble, IsCancelable, IsComposed, MonotonicTime timestamp, RefPtr<WindowProxy>&&, int detail,
        const IntPoint& screenLocation, const IntPoint& windowLocation, double movementX, double movementY, OptionSet<Modifier>, MouseButton, unsigned short buttons,
        EventTarget* relatedTarget, double force, SyntheticClickType, const Vector<Ref<MouseEvent>>& coalescedEvents, const Vector<Ref<MouseEvent>>& predictedEvents, IsSimulated, IsTrusted);

    MouseEvent(enum EventInterfaceType, const AtomString& type, CanBubble, IsCancelable, IsComposed, MonotonicTime timestamp, RefPtr<WindowProxy>&&, int detail,
        const IntPoint& screenLocation, const IntPoint& clientLocation, double movementX, double movementY, OptionSet<Modifier>, MouseButton, unsigned short buttons,
        SyntheticClickType, EventTarget* relatedTarget);

    MouseEvent(enum EventInterfaceType, const AtomString& type, const MouseEventInit&, IsTrusted);

    MouseEvent(enum EventInterfaceType);

private:
    bool isMouseEvent() const final;

    void setRelatedTarget(RefPtr<EventTarget>&& relatedTarget) final { m_relatedTarget = WTFMove(relatedTarget); }

    int16_t m_button { enumToUnderlyingType(MouseButton::Left) };
    unsigned short m_buttons { 0 };
    SyntheticClickType m_syntheticClickType { SyntheticClickType::NoTap };
    bool m_buttonDown { false };
    RefPtr<EventTarget> m_relatedTarget;
    double m_force { 0 };
    Vector<Ref<MouseEvent>> m_coalescedEvents;
    Vector<Ref<MouseEvent>> m_predictedEvents;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_EVENT(MouseEvent)
