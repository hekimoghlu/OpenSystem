/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 22, 2025.
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

#include "SVGRenderStyleDefs.h"
#include <wtf/Noncopyable.h>

namespace WebCore {

class FontCascade;
class RenderObject;
class SVGElement;
class SVGRenderStyle;
class SVGTextMetrics;

// Helper class used by SVGTextLayoutEngine to handle 'alignment-baseline' / 'dominant-baseline' and 'baseline-shift'.
class SVGTextLayoutEngineBaseline {
    WTF_MAKE_NONCOPYABLE(SVGTextLayoutEngineBaseline);
public:
    SVGTextLayoutEngineBaseline(const FontCascade&);

    float calculateBaselineShift(const SVGRenderStyle&, SVGElement* context) const;
    float calculateAlignmentBaselineShift(bool isVerticalText, const RenderObject& textRenderer) const;
    float calculateGlyphOrientationAngle(bool isVerticalText, const SVGRenderStyle&, const UChar& character) const;
    float calculateGlyphAdvanceAndOrientation(bool isVerticalText, SVGTextMetrics&, float angle, float& xOrientationShift, float& yOrientationShift) const;

private:
    AlignmentBaseline dominantBaselineToAlignmentBaseline(bool isVerticalText, const RenderObject* textRenderer) const;

    const FontCascade& m_font;
};

} // namespace WebCore
