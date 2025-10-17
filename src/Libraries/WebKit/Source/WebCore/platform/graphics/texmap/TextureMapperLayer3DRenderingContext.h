/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 15, 2023.
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

#include "FloatPolygon3D.h"
#include "FloatQuad.h"
#include "TextureMapperLayer.h"
#include <wtf/HashSet.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ClipPath;
class FloatPlane3D;
class TextureMapper;

class TextureMapperLayer3DRenderingContext final {
    WTF_MAKE_TZONE_ALLOCATED(TextureMapperLayerPreserves3DContext);
public:
    void paint(TextureMapper&, const Vector<TextureMapperLayer*>&,
        const std::function<void(TextureMapperLayer*, const ClipPath&)>&);

private:
    enum class LayerPosition {
        InFront,
        Behind,
        Coplanar,
        Intersecting
    };

    struct BoundingBox final {
        FloatPoint3D min;
        FloatPoint3D max;
    };

    struct Layer final {
        FloatPolygon3D geometry;
        BoundingBox boundingBox;
        TextureMapperLayer* textureMapperLayer { nullptr };
        bool isSplitted { false };
        unsigned clipVertexBufferOffset { 0 };
    };

    struct LayerNode final {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;

        explicit LayerNode(Layer&& layer)
        {
            layers.append(WTFMove(layer));
        }

        const Layer& firstLayer() const  { return layers[0]; }

        Vector<Layer> layers;
        std::unique_ptr<LayerNode> frontNode;
        std::unique_ptr<LayerNode> backNode;
    };

    using SweepAndPrunePairs = HashSet<std::pair<size_t, size_t>>;

    static BoundingBox computeBoundingBox(const FloatPolygon3D&);
    static SweepAndPrunePairs sweepAndPrune(const Vector<Layer>&);
    static void buildTree(LayerNode&, Deque<Layer>&);
    static void traverseTree(LayerNode&, const std::function<void(LayerNode&)>&);
    static LayerPosition classifyLayer(const Layer&, const FloatPlane3D&);
};

} // namespace WebCore
