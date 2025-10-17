/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 16, 2022.
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
#include "GraphicsLayerTransform.h"

namespace WebCore {

GraphicsLayerTransform::GraphicsLayerTransform()
    : m_anchorPoint(0.5, 0.5, 0)
    , m_flattening(true)
    , m_dirty(false) // false by default since all default values would be combined as the identity matrix
    , m_childrenDirty(false)
{
}

void GraphicsLayerTransform::setPosition(const FloatPoint& position)
{
    if (m_position == position)
        return;
    m_position = position;
    m_dirty = true;
}

void GraphicsLayerTransform::setSize(const FloatSize& size)
{
    if (m_size == size)
        return;
    m_size = size;
    m_dirty = true;
}

void GraphicsLayerTransform::setAnchorPoint(const FloatPoint3D& anchorPoint)
{
    if (m_anchorPoint == anchorPoint)
        return;
    m_anchorPoint = anchorPoint;
    m_dirty = true;
}

void GraphicsLayerTransform::setFlattening(bool flattening)
{
    if (m_flattening == flattening)
        return;
    m_flattening = flattening;
    m_dirty = true;
}

void GraphicsLayerTransform::setLocalTransform(const TransformationMatrix& transform)
{
    if (m_local == transform)
        return;
    m_local = transform;
    m_dirty = true;
}

void GraphicsLayerTransform::setChildrenTransform(const TransformationMatrix& transform)
{
    if (m_children == transform)
        return;
    m_children = transform;
    m_dirty = true;
}

const TransformationMatrix& GraphicsLayerTransform::combined() const
{
    ASSERT(!m_dirty);
    return m_combined;
}

const TransformationMatrix& GraphicsLayerTransform::combinedForChildren() const
{
    ASSERT(!m_dirty);
    if (m_childrenDirty)
        combineTransformsForChildren();
    return m_combinedForChildren;
}

void GraphicsLayerTransform::combineTransforms(const TransformationMatrix& parentTransform)
{
    float originX = m_anchorPoint.x() * m_size.width();
    float originY = m_anchorPoint.y() * m_size.height();
    m_combined = parentTransform;
    m_combined
        .translate3d(originX + m_position.x(), originY + m_position.y(), m_anchorPoint.z())
        .multiply(m_local);

    // The children transform will take it from here, if it gets used.
    m_combinedForChildren = m_combined;
    m_combined.translate3d(-originX, -originY, -m_anchorPoint.z());

    m_dirty = false;
    m_childrenDirty = true;
}

void GraphicsLayerTransform::combineTransformsForChildren() const
{
    ASSERT(!m_dirty);
    ASSERT(m_childrenDirty);

    float originX = m_anchorPoint.x() * m_size.width();
    float originY = m_anchorPoint.y() * m_size.height();

    // In case a parent had preserves3D and this layer has not, flatten our children.
    if (m_flattening)
        m_combinedForChildren = m_combinedForChildren.to2dTransform();
    m_combinedForChildren.multiply(m_children);
    m_combinedForChildren.translate3d(-originX, -originY, -m_anchorPoint.z());

    m_childrenDirty = false;
}

}
