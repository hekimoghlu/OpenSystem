/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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

#include "SVGElement.h"
#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class SVGVisitedElementTracking {
public:
    using VisitedSet = WeakHashSet<SVGElement, WeakPtrImplWithEventTargetData>;

    SVGVisitedElementTracking(VisitedSet& visitedSet)
        : m_visitedElements(visitedSet)
    {
    }

    ~SVGVisitedElementTracking() = default;

    bool isEmpty() const { return m_visitedElements.isEmptyIgnoringNullReferences(); }
    bool isVisiting(const SVGElement& element) { return m_visitedElements.contains(element); }

    class Scope {
    public:
        Scope(SVGVisitedElementTracking& tracking, const SVGElement& element)
            : m_tracking(tracking)
            , m_element(element)
        {
            m_tracking.addUnique(element);
        }

        ~Scope()
        {
            if (RefPtr element = m_element.get())
                m_tracking.removeUnique(*element);
        }

    private:
        SVGVisitedElementTracking& m_tracking;
        WeakPtr<SVGElement, WeakPtrImplWithEventTargetData> m_element;
    };

private:
    friend class Scope;

    void addUnique(const SVGElement& element)
    {
        auto result = m_visitedElements.add(element);
        ASSERT_UNUSED(result, result.isNewEntry);
    }

    void removeUnique(const SVGElement& element)
    {
        bool result = m_visitedElements.remove(element);
        ASSERT_UNUSED(result, result);
    }

    VisitedSet& m_visitedElements;
};

}; // namespace WebCore
