/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 31, 2022.
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

#include "InlineIteratorTextBox.h"
#include "LayoutRect.h"
#include "RegionContext.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class RenderBox;
class RenderBoxModelObject;
class RenderText;
class RenderView;

class AccessibilityRegionContext final : public RegionContext {
    WTF_MAKE_TZONE_ALLOCATED(AccessibilityRegionContext);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(AccessibilityRegionContext);
public:
    AccessibilityRegionContext() = default;
    virtual ~AccessibilityRegionContext();

    bool isAccessibilityRegionContext() const final { return true; }

    // This group of methods takes paint-time geometry and uses it directly.
    void takeBounds(const RenderBox&, LayoutPoint /* paintOffset */);
    void takeBounds(const RenderBox&, FloatRect /* paintRect */);
    void takeBounds(const RenderInline* renderInline, LayoutRect&& paintRect)
    {
        if (renderInline)
            takeBounds(*renderInline, WTFMove(paintRect));
    };
    void takeBounds(const RenderInline&, LayoutRect&& /* paintRect */);
    void takeBounds(const RenderText&, FloatRect /* paintRect */);
    void takeBounds(const RenderView&, LayoutPoint&& /* paintOffset */);

    // This group of methods serves only as a notification that the given object is
    // being painted. From there, we construct the geometry we need ourselves
    // (cheaply, i.e. by combining already-computed geometry how we need it).
    void onPaint(const ScrollView&);

private:
    void takeBoundsInternal(const RenderBoxModelObject&, IntRect&& /* paintRect */);

    // Maps the given rect using the current transform and clip stack.
    // Assumes `rect` is in page-absolute coordinate space (because the clips being applied are).
    template<typename RectT>
    FloatRect mapRect(RectT&& rect)
    {
        bool hasTransform = m_transformStack.size();
        bool hasClip = m_clipStack.size();
        if (!hasTransform && !hasClip)
            return rect;

        FloatRect mappedRect = rect;
        if (hasTransform)
            mappedRect = m_transformStack.last().mapRect(mappedRect);
        if (hasClip)
            mappedRect.intersect(m_clipStack.last());

        return mappedRect;
    }

    SingleThreadWeakHashMap<RenderText, FloatRect> m_accumulatedRenderTextRects;
}; // class AccessibilityRegionContext

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AccessibilityRegionContext)
    static bool isType(const WebCore::RegionContext& regionContext) { return regionContext.isAccessibilityRegionContext(); }
SPECIALIZE_TYPE_TRAITS_END()
