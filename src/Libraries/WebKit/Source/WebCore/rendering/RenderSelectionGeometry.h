/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 25, 2025.
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

#include "GapRects.h"
#include "RenderBlock.h"
#include "RenderObject.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class RenderSelectionGeometryBase {
    WTF_MAKE_TZONE_ALLOCATED(RenderSelectionGeometryBase);
    WTF_MAKE_NONCOPYABLE(RenderSelectionGeometryBase);
public:
    explicit RenderSelectionGeometryBase(RenderObject& renderer);
    const RenderLayerModelObject* repaintContainer() const { return m_repaintContainer; }
    RenderObject::HighlightState state() const { return m_state; }

protected:
    void repaintRectangle(const LayoutRect& repaintRect);

    RenderObject& m_renderer;
    const RenderLayerModelObject* m_repaintContainer;

private:
    RenderObject::HighlightState m_state;
};

// This struct is used when the selection changes to cache the old and new state of the selection for each RenderObject.
class RenderSelectionGeometry : public RenderSelectionGeometryBase {
    WTF_MAKE_TZONE_ALLOCATED(RenderSelectionGeometry);
public:
    RenderSelectionGeometry(RenderObject& renderer, bool clipToVisibleContent);

    void repaint();
    const Vector<FloatQuad>& collectedSelectionQuads() const { return m_collectedSelectionQuads; }
    LayoutRect rect() const { return m_rect; }

private:
    Vector<FloatQuad> m_collectedSelectionQuads; // relative to repaint container
    LayoutRect m_rect; // relative to repaint container
};


// This struct is used when the selection changes to cache the old and new state of the selection for each RenderBlock.
class RenderBlockSelectionGeometry : public RenderSelectionGeometryBase {
    WTF_MAKE_TZONE_ALLOCATED(RenderBlockSelectionGeometry);
public:
    explicit RenderBlockSelectionGeometry(RenderBlock& renderer);

    void repaint();
    GapRects rects() const { return m_rects; }

private:
    GapRects m_rects; // relative to repaint container
};

} // namespace WebCore
