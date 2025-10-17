/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 1, 2022.
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
#include "EventContext.h"

#include "Document.h"
#include "EventNames.h"
#include "FocusEvent.h"
#include "HTMLFieldSetElement.h"
#include "HTMLFormElement.h"
#include "LocalDOMWindow.h"
#include "MouseEvent.h"
#include "TouchEvent.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(EventContext);

void EventContext::handleLocalEvents(Event& event, EventInvokePhase phase) const
{
    event.setTarget(m_target.copyRef());
    event.setCurrentTarget(m_currentTarget.copyRef(), m_currentTargetIsInShadowTree);

    if (m_relatedTargetIsSet) {
        ASSERT(!m_relatedTarget || isMouseOrFocusEventContext() || isWindowContext());
        event.setRelatedTarget(m_relatedTarget.copyRef());
    }

#if ENABLE(TOUCH_EVENTS)
    if (m_type == Type::Touch) {

#if ASSERT_ENABLED
        auto checkReachability = [&](const Ref<TouchList>& touchList) {
            size_t length = touchList->length();
            for (size_t i = 0; i < length; ++i)
                ASSERT(!isUnreachableNode(downcast<Node>(touchList->item(i)->target())));
        };
        checkReachability(*m_touches);
        checkReachability(*m_targetTouches);
        checkReachability(*m_changedTouches);
#endif

        auto& touchEvent = downcast<TouchEvent>(event);
        touchEvent.setTouches(m_touches.get());
        touchEvent.setTargetTouches(m_targetTouches.get());
        touchEvent.setChangedTouches(m_changedTouches.get());
    }
#endif

    if (!m_node || UNLIKELY(m_type == Type::Window)) {
        protectedCurrentTarget()->fireEventListeners(event, phase);
        return;
    }

    if (UNLIKELY(m_contextNodeIsFormElement)) {
        ASSERT(is<HTMLFormElement>(*m_node));
        auto& eventNames = WebCore::eventNames();
        if ((event.type() == eventNames.submitEvent || event.type() == eventNames.resetEvent)
            && event.eventPhase() != Event::CAPTURING_PHASE && event.target() != m_node && is<Node>(event.target())) {
            event.stopPropagation();
            return;
        }
    }

    if (!m_node->hasEventTargetData())
        return;

    protectedNode()->fireEventListeners(event, phase);
}

#if ENABLE(TOUCH_EVENTS)

void EventContext::initializeTouchLists()
{
    m_touches = TouchList::create();
    m_targetTouches = TouchList::create();
    m_changedTouches = TouchList::create();
}

#endif // ENABLE(TOUCH_EVENTS)

#if ASSERT_ENABLED

bool EventContext::isUnreachableNode(EventTarget* target) const
{
    // FIXME: Checks also for SVG elements.
    auto* node = dynamicDowncast<Node>(target);
    return node && !node->isSVGElement() && m_node && m_node->isClosedShadowHidden(*node);
}

#endif

}
