/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 10, 2023.
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

#include "ClipRect.h"

namespace WebCore {

class LayerFragment {
public:
    LayerFragment() = default;
    
    void setRects(const LayoutRect& bounds, const ClipRect& background, const ClipRect& foreground, const std::optional<LayoutRect>& bbox)
    {
        layerBounds = bounds;
        backgroundRect = background;
        foregroundRect = foreground;
        boundingBox = bbox;
    }
    
    void moveBy(const LayoutPoint& offset)
    {
        layerBounds.moveBy(offset);
        backgroundRect.moveBy(offset);
        foregroundRect.moveBy(offset);
        paginationClip.moveBy(offset);
        if (boundingBox)
            boundingBox->moveBy(offset);
    }
    
    void intersect(const LayoutRect& rect)
    {
        backgroundRect.intersect(rect);
        foregroundRect.intersect(rect);
        if (boundingBox)
            boundingBox->intersect(rect);
    }
    
    void intersect(const ClipRect& clipRect)
    {
        backgroundRect.intersect(clipRect);
        foregroundRect.intersect(clipRect);
    }

    bool shouldPaintContent = false;
    std::optional<LayoutRect> boundingBox;

    LayoutRect layerBounds;
    ClipRect backgroundRect;
    ClipRect foregroundRect;
    
    // Unique to paginated fragments. The physical translation to apply to shift the layer when painting/hit-testing.
    LayoutSize paginationOffset;
    
    // Also unique to paginated fragments. An additional clip that applies to the layer. It is in layer-local
    // (physical) coordinates.
    LayoutRect paginationClip;
};

typedef Vector<LayerFragment, 1> LayerFragments;

} // namespace WebCore
