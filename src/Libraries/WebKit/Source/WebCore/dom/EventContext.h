/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 11, 2025.
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

#include "HTMLFormElement.h"
#include "TouchList.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class EventContext {
    WTF_MAKE_TZONE_ALLOCATED(EventContext);
public:
    using EventInvokePhase = EventTarget::EventInvokePhase;

    enum class Type : uint8_t {
        Normal = 0,
        MouseOrFocus,
        Touch,
        Window,
    };

    EventContext(Type, Node*, EventTarget* currentTarget, EventTarget* origin, int closedShadowDepth);
    EventContext(Type, Node&, Node* currentTarget, EventTarget* origin, int closedShadowDepth);
    ~EventContext() = default;

    Node* node() const { return m_node.get(); }
    RefPtr<Node> protectedNode() const { return m_node; }
    EventTarget* currentTarget() const { return m_currentTarget.get(); }
    RefPtr<EventTarget> protectedCurrentTarget() const { return m_currentTarget; }
    bool isCurrentTargetInShadowTree() const { return m_currentTargetIsInShadowTree; }
    EventTarget* target() const { return m_target.get(); }
    int closedShadowDepth() const { return m_closedShadowDepth; }

    void handleLocalEvents(Event&, EventInvokePhase) const;

    bool isNormalEventContext() const { return m_type == Type::Normal; }
    bool isMouseOrFocusEventContext() const { return m_type == Type::MouseOrFocus; }
    bool isTouchEventContext() const { return m_type == Type::Touch; }
    bool isWindowContext() const { return m_type == Type::Window; }

    Node* relatedTarget() const { return m_relatedTarget.get(); }
    void setRelatedTarget(RefPtr<Node>&&);

#if ENABLE(TOUCH_EVENTS)
    enum class TouchListType : uint8_t { Touches, TargetTouches, ChangedTouches };
    TouchList& touchList(TouchListType);
#endif

private:
    inline EventContext(Type, Node* currentNode, RefPtr<EventTarget>&& currentTarget, EventTarget* origin, int closedShadowDepth, bool currentTargetIsInShadowTree = false);

#if ENABLE(TOUCH_EVENTS)
    void initializeTouchLists();
#endif

#if ASSERT_ENABLED
    bool isUnreachableNode(EventTarget*) const;
#endif

    RefPtr<Node> m_node;
    RefPtr<EventTarget> m_currentTarget;
    RefPtr<EventTarget> m_target;
    RefPtr<Node> m_relatedTarget;
#if ENABLE(TOUCH_EVENTS)
    RefPtr<TouchList> m_touches;
    RefPtr<TouchList> m_targetTouches;
    RefPtr<TouchList> m_changedTouches;
#endif
    int m_closedShadowDepth { 0 };
    bool m_currentTargetIsInShadowTree { false };
    bool m_contextNodeIsFormElement { false };
    bool m_relatedTargetIsSet { false };
    Type m_type { Type::Normal };
};

inline EventContext::EventContext(Type type, Node* node, RefPtr<EventTarget>&& currentTarget, EventTarget* origin, int closedShadowDepth, bool currentTargetIsInShadowTree)
    : m_node { node }
    , m_currentTarget { WTFMove(currentTarget) }
    , m_target { origin }
    , m_closedShadowDepth { closedShadowDepth }
    , m_currentTargetIsInShadowTree { currentTargetIsInShadowTree }
    , m_type { type }
{
    ASSERT(!isUnreachableNode(m_target.get()));
#if ENABLE(TOUCH_EVENTS)
    if (m_type == Type::Touch)
        initializeTouchLists();
#else
    ASSERT(m_type != Type::Touch);
#endif
}

inline EventContext::EventContext(Type type, Node* node, EventTarget* currentTarget, EventTarget* origin, int closedShadowDepth)
    : EventContext(type, node, RefPtr { currentTarget }, origin, closedShadowDepth)
{
}

// This variant avoids calling EventTarget::ref() which is a virtual function call.
inline EventContext::EventContext(Type type, Node& node, Node* currentTarget, EventTarget* origin, int closedShadowDepth)
    : EventContext(type, &node, RefPtr { currentTarget }, origin, closedShadowDepth, currentTarget && currentTarget->isInShadowTree())
{
    m_contextNodeIsFormElement = is<HTMLFormElement>(node);
}

inline void EventContext::setRelatedTarget(RefPtr<Node>&& relatedTarget)
{
    ASSERT(!isUnreachableNode(relatedTarget.get()));
    m_relatedTarget = WTFMove(relatedTarget);
    m_relatedTargetIsSet = true;
}

#if ENABLE(TOUCH_EVENTS)

inline TouchList& EventContext::touchList(TouchListType type)
{
    switch (type) {
    case TouchListType::Touches:
        return *m_touches;
    case TouchListType::TargetTouches:
        return *m_targetTouches;
    case TouchListType::ChangedTouches:
        return *m_changedTouches;
    }
    ASSERT_NOT_REACHED();
    return *m_touches;
}

#endif

} // namespace WebCore
