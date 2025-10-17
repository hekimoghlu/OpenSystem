/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 5, 2023.
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
#include "RenderGeometryMap.h"

namespace WTF {
class TextStream;
}

namespace WebCore {

class OverflowAwareOverlapContainer;
class OverlapMapContainer;
class RenderLayer;

class LayerOverlapMap {
    WTF_MAKE_NONCOPYABLE(LayerOverlapMap);
public:
    LayerOverlapMap(const RenderLayer& rootLayer);
    ~LayerOverlapMap();
    
    struct LayerAndBounds {
        RenderLayer& layer;
        LayoutRect bounds;
    };

    using LayerAndBoundsVector = Vector<LayerAndBounds, 2>;

    void add(const RenderLayer&, const LayoutRect&, const LayerAndBoundsVector& enclosingClippingLayers);
    bool overlapsLayers(const RenderLayer&, const LayoutRect&, const LayerAndBoundsVector& enclosingClippingLayers) const;
    bool isEmpty() const { return m_isEmpty; }

    void pushCompositingContainer(const RenderLayer&);
    void popCompositingContainer(const RenderLayer&);

    void pushSpeculativeCompositingContainer(const RenderLayer&);
    void confirmSpeculativeCompositingContainer();
    bool maybePopSpeculativeCompositingContainer();

    const RenderGeometryMap& geometryMap() const { return m_geometryMap; }
    RenderGeometryMap& geometryMap() { return m_geometryMap; }

    const Vector<std::unique_ptr<OverlapMapContainer>>& overlapStack() const { return m_overlapStack; }

private:
    Vector<std::unique_ptr<OverlapMapContainer>> m_overlapStack;
    Vector<std::unique_ptr<OverlapMapContainer>> m_speculativeOverlapStack;
    RenderGeometryMap m_geometryMap;
    const RenderLayer& m_rootLayer;
    bool m_isEmpty { true };
};

TextStream& operator<<(TextStream&, const LayerOverlapMap&);

} // namespace WebCore
