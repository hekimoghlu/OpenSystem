/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 5, 2023.
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

#include "EventContext.h"
#include "PseudoElement.h"
#include "SVGElement.h"
#include "SVGUseElement.h"
#include <wtf/Forward.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class EventPath;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::EventPath> : std::true_type { };
}

namespace WebCore {

class Touch;

class EventPath : public CanMakeSingleThreadWeakPtr<EventPath> {
public:
    EventPath(Node& origin, Event&);
    explicit EventPath(const Vector<EventTarget*>&);
    explicit EventPath(EventTarget&);

    bool isEmpty() const { return m_path.isEmpty(); }
    size_t size() const { return m_path.size(); }
    const EventContext& contextAt(size_t i) const { return m_path[i]; }
    EventContext& contextAt(size_t i) { return m_path[i]; }

    void adjustForDisabledFormControl();

    Vector<Ref<EventTarget>> computePathUnclosedToTarget(const EventTarget&) const;

    static Node* eventTargetRespectingTargetRules(Node&);

private:
    void buildPath(Node& origin, Event&);
    void setRelatedTarget(Node& origin, Node&);

#if ENABLE(TOUCH_EVENTS)
    void retargetTouch(EventContext::TouchListType, const Touch&);
    void retargetTouchList(EventContext::TouchListType, const TouchList*);
    void retargetTouchLists(const TouchEvent&);
#endif

    Vector<EventContext, 32> m_path;
};

inline Node* EventPath::eventTargetRespectingTargetRules(Node& referenceNode)
{
    if (auto* pseudoElement = dynamicDowncast<PseudoElement>(referenceNode))
        return pseudoElement->hostElement();

    // Events sent to elements inside an SVG use element's shadow tree go to the use element.
    if (auto* svgElement = dynamicDowncast<SVGElement>(referenceNode)) {
        if (auto useElement = svgElement->correspondingUseElement())
            return useElement.get();
    }

    return &referenceNode;
}

} // namespace WebCore
