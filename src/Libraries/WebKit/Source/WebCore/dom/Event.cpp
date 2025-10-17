/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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
#include "Event.h"

#include "Document.h"
#include "EventNames.h"
#include "EventPath.h"
#include "EventTarget.h"
#include "InspectorInstrumentation.h"
#include "LocalDOMWindow.h"
#include "Performance.h"
#include "UserGestureIndicator.h"
#include "WorkerGlobalScope.h"
#include <wtf/HexNumber.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(Event);

ALWAYS_INLINE Event::Event(MonotonicTime createTime, enum EventInterfaceType eventInterface, const AtomString& type, IsTrusted isTrusted, CanBubble canBubble, IsCancelable cancelable, IsComposed composed)
    : m_isInitialized { !type.isNull() }
    , m_canBubble { canBubble == CanBubble::Yes }
    , m_cancelable { cancelable == IsCancelable::Yes }
    , m_composed { composed == IsComposed::Yes }
    , m_propagationStopped { false }
    , m_immediatePropagationStopped { false }
    , m_wasCanceled { false }
    , m_defaultHandled { false }
    , m_isDefaultEventHandlerIgnored { false }
    , m_isTrusted { isTrusted == IsTrusted::Yes }
    , m_isExecutingPassiveEventListener { false }
    , m_currentTargetIsInShadowTree { false }
    , m_isAutofillEvent { false }
    , m_eventPhase { NONE }
    , m_eventInterface(enumToUnderlyingType(eventInterface))
    , m_type { type }
    , m_createTime { createTime }
{
    ASSERT(m_eventInterface == enumToUnderlyingType(eventInterface));
}

Event::Event(enum EventInterfaceType eventInterface, IsTrusted isTrusted)
    : Event { MonotonicTime::now(), eventInterface, { }, isTrusted, CanBubble::No, IsCancelable::No, IsComposed::No }
{
}

Event::Event(enum EventInterfaceType eventInterface, const AtomString& eventType, CanBubble canBubble, IsCancelable isCancelable, IsComposed isComposed)
    : Event { MonotonicTime::now(), eventInterface, eventType, IsTrusted::Yes, canBubble, isCancelable, isComposed }
{
    ASSERT(!eventType.isNull());
}

Event::Event(enum EventInterfaceType eventInterface, const AtomString& eventType, CanBubble canBubble, IsCancelable cancelable, IsComposed composed, MonotonicTime timestamp, IsTrusted isTrusted)
    : Event(timestamp, eventInterface, eventType, isTrusted, canBubble, cancelable, composed)
{
}

Event::Event(enum EventInterfaceType eventInterface, const AtomString& eventType, const EventInit& initializer, IsTrusted isTrusted)
    : Event { MonotonicTime::now(), eventInterface, eventType, isTrusted,
        initializer.bubbles ? CanBubble::Yes : CanBubble::No,
        initializer.cancelable ? IsCancelable::Yes : IsCancelable::No,
        initializer.composed ? IsComposed::Yes : IsComposed::No }
{
    ASSERT(!eventType.isNull());
    m_isConstructedFromInitializer = true;
}

Event::~Event() = default;

Ref<Event> Event::create(const AtomString& type, CanBubble canBubble, IsCancelable isCancelable, IsComposed isComposed)
{
    return adoptRef(*new Event(EventInterfaceType::Event, type, canBubble, isCancelable, isComposed));
}

Ref<Event> Event::createForBindings()
{
    return adoptRef(*new Event(EventInterfaceType::Event));
}

Ref<Event> Event::create(const AtomString& type, const EventInit& initializer, IsTrusted isTrusted)
{
    return adoptRef(*new Event(EventInterfaceType::Event, type, initializer, isTrusted));
}

void Event::initEvent(const AtomString& eventTypeArg, bool canBubbleArg, bool cancelableArg)
{
    if (isBeingDispatched())
        return;

    m_isInitialized = true;
    m_propagationStopped = false;
    m_immediatePropagationStopped = false;
    m_wasCanceled = false;
    m_isTrusted = false;
    m_target = nullptr;
    m_type = eventTypeArg;
    m_canBubble = canBubbleArg;
    m_cancelable = cancelableArg;

    m_underlyingEvent = nullptr;
}

void Event::setTarget(RefPtr<EventTarget>&& target)
{
    if (m_target == target)
        return;

    m_target = WTFMove(target);
    if (m_target)
        receivedTarget();
}

RefPtr<EventTarget> Event::protectedCurrentTarget() const
{
    return m_currentTarget;
}

void Event::setCurrentTarget(RefPtr<EventTarget>&& currentTarget, std::optional<bool> isInShadowTree)
{
    m_currentTarget = WTFMove(currentTarget);
    if (isInShadowTree)
        m_currentTargetIsInShadowTree = *isInShadowTree;
    else {
        auto* targetNode = dynamicDowncast<Node>(m_currentTarget.get());
        m_currentTargetIsInShadowTree = targetNode && targetNode->isInShadowTree();
    }
}

void Event::setEventPath(const EventPath& path)
{
    m_eventPath = &path;
}

Vector<Ref<EventTarget>> Event::composedPath() const
{
    if (!m_eventPath)
        return Vector<Ref<EventTarget>>();
    return m_eventPath->computePathUnclosedToTarget(*protectedCurrentTarget());
}

void Event::setUnderlyingEvent(Event* underlyingEvent)
{
    // Prohibit creation of a cycle by doing nothing if a cycle would be created.
    for (Event* event = underlyingEvent; event; event = event->underlyingEvent()) {
        if (event == this)
            return;
    }
    m_underlyingEvent = underlyingEvent;
}

DOMHighResTimeStamp Event::timeStampForBindings(ScriptExecutionContext& context) const
{
    RefPtr<Performance> performance;
    if (auto* globalScope = dynamicDowncast<WorkerGlobalScope>(context))
        performance = &globalScope->performance();
    else if (RefPtr window = downcast<Document>(context).domWindow())
        performance = &window->performance();

    if (!performance)
        return 0;

    return std::max(performance->relativeTimeFromTimeOriginInReducedResolution(m_createTime), 0.);
}

void Event::resetBeforeDispatch()
{
    m_defaultHandled = false;
}

void Event::resetAfterDispatch()
{
    m_eventPath = nullptr;
    setCurrentTarget(nullptr);
    m_eventPhase = NONE;
    m_propagationStopped = false;
    m_immediatePropagationStopped = false;

    InspectorInstrumentation::eventDidResetAfterDispatch(*this);
}

String Event::debugDescription() const
{
    return makeString(type(), " phase "_s, eventPhase(), bubbles() ? " bubbles "_s : " "_s, cancelable() ? "cancelable "_s : " "_s, "0x"_s, hex(reinterpret_cast<uintptr_t>(this), Lowercase));
}

TextStream& operator<<(TextStream& ts, const Event& event)
{
    ts << event.debugDescription();
    return ts;
}

} // namespace WebCore
