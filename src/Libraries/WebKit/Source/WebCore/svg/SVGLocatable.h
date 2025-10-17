/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 5, 2024.
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
#include "ExceptionOr.h"

namespace WebCore {

class FloatRect;
class SVGElement;
class SVGMatrix;

class SVGLocatable {
public:
    virtual ~SVGLocatable() = default;

    // 'SVGLocatable' functions
    virtual SVGElement* nearestViewportElement() const = 0;
    virtual SVGElement* farthestViewportElement() const = 0;

    enum StyleUpdateStrategy { AllowStyleUpdate, DisallowStyleUpdate };
    
    virtual FloatRect getBBox(StyleUpdateStrategy) = 0;
    virtual AffineTransform getCTM(StyleUpdateStrategy) = 0;
    virtual AffineTransform getScreenCTM(StyleUpdateStrategy) = 0;

    static SVGElement* nearestViewportElement(const SVGElement*);
    static SVGElement* farthestViewportElement(const SVGElement*);

    enum CTMScope {
        NearestViewportScope, // Used for getCTM()
        ScreenScope // Used for getScreenCTM()
    };

protected:
    virtual AffineTransform localCoordinateSpaceTransform(SVGLocatable::CTMScope) const { return AffineTransform(); }

    static FloatRect getBBox(SVGElement*, StyleUpdateStrategy);
    static AffineTransform computeCTM(SVGElement*, CTMScope, StyleUpdateStrategy);
};

} // namespace WebCore
