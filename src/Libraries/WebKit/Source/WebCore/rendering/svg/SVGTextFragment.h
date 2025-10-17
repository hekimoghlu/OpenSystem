/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 11, 2022.
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

#include "AffineTransform.h"

namespace WebCore {

// A SVGTextFragment describes a text fragment of a RenderSVGInlineText which can be rendered at once.
struct SVGTextFragment {
    SVGTextFragment()
        : characterOffset(0)
        , metricsListOffset(0)
        , length(0)
        , isTextOnPath(false)
        , x(0)
        , y(0)
        , width(0)
        , height(0)
    {
    }

    enum TransformType {
        TransformRespectingTextLength,
        TransformIgnoringTextLength
    };

    void buildFragmentTransform(AffineTransform& result, TransformType type = TransformRespectingTextLength) const
    {
        if (type == TransformIgnoringTextLength) {
            result = transform;
            transformAroundOrigin(result);
            return;
        }

        if (isTextOnPath)
            buildTransformForTextOnPath(result);
        else
            buildTransformForTextOnLine(result);
    }

    // The first rendered character starts at RenderSVGInlineText::characters() + characterOffset.
    unsigned characterOffset;
    unsigned metricsListOffset;
    unsigned length : 31;
    bool isTextOnPath : 1;

    float x;
    float y;
    float width;
    float height;

    // Includes rotation/glyph-orientation-(horizontal|vertical) transforms, as well as orientation related shifts
    // (see SVGTextLayoutEngine, which builds this transformation).
    AffineTransform transform;

    // Contains lengthAdjust related transformations, which are not allowd to influence the SVGTextQuery code.
    AffineTransform lengthAdjustTransform;

private:
    void transformAroundOrigin(AffineTransform& result) const
    {
        // Returns (translate(x, y) * result) * translate(-x, -y).
        result.setE(result.e() + x);
        result.setF(result.f() + y);
        result.translate(-x, -y);
    }

    void buildTransformForTextOnPath(AffineTransform& result) const
    {
        // For text-on-path layout, multiply the transform with the lengthAdjustTransform before orienting the resulting transform.
        result = lengthAdjustTransform.isIdentity() ? transform : transform * lengthAdjustTransform;
        if (!result.isIdentity())
            transformAroundOrigin(result);
    }

    void buildTransformForTextOnLine(AffineTransform& result) const
    {
        // For text-on-line layout, orient the transform first, then multiply the lengthAdjustTransform with the oriented transform.
        if (transform.isIdentity()) {
            result = lengthAdjustTransform;
            return;
        }

        result = transform;
        transformAroundOrigin(result);

        if (!lengthAdjustTransform.isIdentity())
            result = lengthAdjustTransform * result;
    }
};

} // namespace WebCore
