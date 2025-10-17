/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 10, 2022.
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

#include "FloatPoint.h"
#include "FloatQuad.h"
#include "LayoutSize.h"
#include "RenderObject.h"
#include "TransformationMatrix.h"
#include <memory>

namespace WebCore {

class RenderFragmentedFlow;
class RenderLayer;
class RenderLayerModelObject;
class RenderView;
class TransformState;

// Stores data about how to map from one renderer to its container.
struct RenderGeometryMapStep {
    RenderGeometryMapStep(const RenderGeometryMapStep& o)
        : m_renderer(o.m_renderer)
        , m_offset(o.m_offset)
        , m_accumulatingTransform(o.m_accumulatingTransform)
        , m_isNonUniform(o.m_isNonUniform)
        , m_isFixedPosition(o.m_isFixedPosition)
        , m_hasTransform(o.m_hasTransform)
    {
        ASSERT(!o.m_transform);
    }
    RenderGeometryMapStep(const RenderObject* renderer, bool accumulatingTransform, bool isNonUniform, bool isFixedPosition, bool hasTransform)
        : m_renderer(renderer)
        , m_accumulatingTransform(accumulatingTransform)
        , m_isNonUniform(isNonUniform)
        , m_isFixedPosition(isFixedPosition)
        , m_hasTransform(hasTransform)
    {
    }
    const RenderObject* m_renderer;
    LayoutSize m_offset;
    std::unique_ptr<TransformationMatrix> m_transform; // Includes offset if non-null.
    bool m_accumulatingTransform;
    bool m_isNonUniform; // Mapping depends on the input point, e.g. because of CSS columns.
    bool m_isFixedPosition;
    bool m_hasTransform;
};

// Can be used while walking the Renderer tree to cache data about offsets and transforms.
class RenderGeometryMap {
    WTF_MAKE_NONCOPYABLE(RenderGeometryMap);
public:
    explicit RenderGeometryMap(OptionSet<MapCoordinatesMode> = UseTransforms);
    ~RenderGeometryMap();

    OptionSet<MapCoordinatesMode> mapCoordinatesFlags() const { return m_mapCoordinatesFlags; }

    FloatPoint absolutePoint(const FloatPoint& p) const
    {
        return mapToContainer(p, nullptr);
    }

    FloatRect absoluteRect(const FloatRect& rect) const
    {
        return mapToContainer(rect, nullptr).boundingBox();
    }

    // Map to a container. Will assert that the container has been pushed onto this map.
    // A null container maps through the RenderView (including its scale transform, if any).
    // If the container is the RenderView, the scroll offset is applied, but not the scale.
    FloatPoint mapToContainer(const FloatPoint&, const RenderLayerModelObject*) const;
    FloatQuad mapToContainer(const FloatRect&, const RenderLayerModelObject*) const;
    
    // Called by code walking the renderer or layer trees.
    void pushMappingsToAncestor(const RenderLayer*, const RenderLayer* ancestorLayer, bool respectTransforms = true);
    void popMappingsToAncestor(const RenderLayer*);
    void pushMappingsToAncestor(const RenderObject*, const RenderLayerModelObject* ancestorRenderer);
    void popMappingsToAncestor(const RenderLayerModelObject*);
    
    // The following methods should only be called by renderers inside a call to pushMappingsToAncestor().

    // Push geometry info between this renderer and some ancestor. The ancestor must be its container() or some
    // stacking context between the renderer and its container.
    void push(const RenderObject*, const LayoutSize&, bool accumulatingTransform = false, bool isNonUniform = false, bool isFixedPosition = false, bool hasTransform = false);
    void push(const RenderObject*, const TransformationMatrix&, bool accumulatingTransform = false, bool isNonUniform = false, bool isFixedPosition = false, bool hasTransform = false);

    // RenderView gets special treatment, because it applies the scroll offset only for elements inside in fixed position.
    void pushView(const RenderView*, const LayoutSize& scrollOffset, const TransformationMatrix* = nullptr);
    void pushRenderFragmentedFlow(const RenderFragmentedFlow*);

private:
    void mapToContainer(TransformState&, const RenderLayerModelObject* container = nullptr) const;

    void stepInserted(const RenderGeometryMapStep&);
    void stepRemoved(const RenderGeometryMapStep&);
    
    bool hasNonUniformStep() const { return m_nonUniformStepsCount; }
    bool hasTransformStep() const { return m_transformedStepsCount; }
    bool hasFixedPositionStep() const { return m_fixedStepsCount; }

    typedef Vector<RenderGeometryMapStep, 32> RenderGeometryMapSteps;

    size_t m_insertionPosition { notFound };
    int m_nonUniformStepsCount { 0 };
    int m_transformedStepsCount { 0 };
    int m_fixedStepsCount { 0 };
    RenderGeometryMapSteps m_mapping;
    LayoutSize m_accumulatedOffset;
    OptionSet<MapCoordinatesMode> m_mapCoordinatesFlags;
#if ASSERT_ENABLED
    bool m_accumulatedOffsetMightBeSaturated { false };
#endif
};

} // namespace WebCore

namespace WTF {
// This is required for a struct with std::unique_ptr<>. We know RenderGeometryMapStep is simple enough that
// initializing to 0 and moving with memcpy (and then not destructing the original) will work.
template<> struct VectorTraits<WebCore::RenderGeometryMapStep> : SimpleClassVectorTraits { };
}
