/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 11, 2023.
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

#include "LayoutRect.h"
#include "RenderLayer.h"
#include "ScrollTypes.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WTF {
class TextStream;
}

namespace WebCore {

class ScrollingCoordinator;

struct CompositedClipData {
    CompositedClipData(RenderLayer* layer, const RoundedRect& roundedRect, bool isOverflowScrollEntry)
        : clippingLayer(layer)
        , clipRect(roundedRect)
        , isOverflowScroll(isOverflowScrollEntry)
    {
    }

    friend bool operator==(const CompositedClipData&, const CompositedClipData&) = default;
    
    SingleThreadWeakPtr<RenderLayer> clippingLayer; // For scroller entries, the scrolling layer. For other entries, the most-descendant layer that has a clip.
    RoundedRect clipRect; // In the coordinate system of the RenderLayer that owns the stack.
    bool isOverflowScroll { false };
};


// This class encapsulates the set of layers and their scrolling tree nodes representing clipping in the layer's containing block ancestry,
// but not in its paint order ancestry.
class LayerAncestorClippingStack {
    WTF_MAKE_TZONE_ALLOCATED(LayerAncestorClippingStack);
public:
    LayerAncestorClippingStack(Vector<CompositedClipData>&&);
    ~LayerAncestorClippingStack() = default;

    bool hasAnyScrollingLayers() const;
    
    bool equalToClipData(const Vector<CompositedClipData>&) const;
    bool updateWithClipData(ScrollingCoordinator*, Vector<CompositedClipData>&&);
    
    Vector<CompositedClipData> compositedClipData() const;

    void clear(ScrollingCoordinator*);
    void detachFromScrollingCoordinator(ScrollingCoordinator&);

    void updateScrollingNodeLayers(ScrollingCoordinator&);

    GraphicsLayer* firstLayer() const;
    GraphicsLayer* lastLayer() const;
    std::optional<ScrollingNodeID> lastOverflowScrollProxyNodeID() const;

    struct ClippingStackEntry {
        CompositedClipData clipData;
        Markable<ScrollingNodeID> overflowScrollProxyNodeID; // The node for repositioning the scrolling proxy layer.
        RefPtr<GraphicsLayer> clippingLayer;
        RefPtr<GraphicsLayer> scrollingLayer; // Only present for scrolling entries.

        GraphicsLayer* parentForSublayers() const
        {
            return scrollingLayer ? scrollingLayer.get() : clippingLayer.get();
        }
        
        GraphicsLayer* childForSuperlayers() const
        {
            return clippingLayer.get();
        }
    };

    Vector<ClippingStackEntry>& stack() { return m_stack; }
    const Vector<ClippingStackEntry>& stack() const { return m_stack; }

private:
    // Order is ancestors to descendants.
    Vector<ClippingStackEntry> m_stack;
};

TextStream& operator<<(TextStream&, const LayerAncestorClippingStack&);

} // namespace WebCore
