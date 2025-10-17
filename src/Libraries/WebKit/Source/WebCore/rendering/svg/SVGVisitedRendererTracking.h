/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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

#include "RenderElement.h"
#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class SVGVisitedRendererTracking {
public:
    using VisitedSet = SingleThreadWeakHashSet<RenderElement>;

    SVGVisitedRendererTracking(VisitedSet& visitedSet)
        : m_visitedRenderers(visitedSet)
    {
    }

    ~SVGVisitedRendererTracking() = default;

    bool isEmpty() const { return m_visitedRenderers.isEmptyIgnoringNullReferences(); }
    bool isVisiting(const RenderElement& renderer) { return m_visitedRenderers.contains(renderer); }

    class Scope {
    public:
        Scope(SVGVisitedRendererTracking& tracking, const RenderElement& renderer)
            : m_tracking(tracking)
            , m_renderer(renderer)
        {
            m_tracking.addUnique(renderer);
        }

        ~Scope()
        {
            if (m_renderer)
                m_tracking.removeUnique(*m_renderer);
        }

    private:
        SVGVisitedRendererTracking& m_tracking;
        SingleThreadWeakPtr<RenderElement> m_renderer;
    };

private:
    friend class Scope;

    void addUnique(const RenderElement& renderer)
    {
        auto result = m_visitedRenderers.add(renderer);
        ASSERT_UNUSED(result, result.isNewEntry);
    }

    void removeUnique(const RenderElement& renderer)
    {
        bool result = m_visitedRenderers.remove(renderer);
        ASSERT_UNUSED(result, result);
    }

    VisitedSet& m_visitedRenderers;
};

}; // namespace WebCore
