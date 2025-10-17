/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 23, 2024.
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
#ifndef GraphicsLayerTransform_h
#define GraphicsLayerTransform_h

#include "FloatPoint.h"
#include "FloatPoint3D.h"
#include "FloatSize.h"
#include "TransformationMatrix.h"

namespace WebCore {

class GraphicsLayerTransform {
public:
    WEBCORE_EXPORT GraphicsLayerTransform();
    void setPosition(const FloatPoint&);
    void setSize(const FloatSize&);
    void setAnchorPoint(const FloatPoint3D&);
    void setFlattening(bool);
    void setLocalTransform(const TransformationMatrix&);
    void setChildrenTransform(const TransformationMatrix&);
    const TransformationMatrix& combined() const;
    const TransformationMatrix& combinedForChildren() const;

    void combineTransforms(const TransformationMatrix& parentTransform);

private:
    void combineTransformsForChildren() const;

    FloatPoint3D m_anchorPoint;
    FloatPoint m_position;
    FloatSize m_size;
    bool m_flattening;
    bool m_dirty;
    mutable bool m_childrenDirty;

    TransformationMatrix m_local;
    TransformationMatrix m_children;
    TransformationMatrix m_combined;
    mutable TransformationMatrix m_combinedForChildren;
};

}

#endif // GraphicsLayerTransform_h
