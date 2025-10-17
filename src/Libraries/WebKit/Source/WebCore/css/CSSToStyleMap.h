/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 12, 2022.
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

#include <wtf/Forward.h>

namespace WebCore {

class Animation;
class CSSBorderImageSliceValue;
class CSSBorderImageWidthValue;
class CSSValue;
class FillLayer;
class LengthBox;
class NinePieceImage;
class Quad;
class RenderStyle;
class StyleImage;

enum CSSPropertyID : uint16_t;

struct Length;

namespace Style {
class BuilderState;
}

class CSSToStyleMap {
public:
    explicit CSSToStyleMap(Style::BuilderState&);

    static void mapFillAttachment(CSSPropertyID, FillLayer&, const CSSValue&);
    static void mapFillClip(CSSPropertyID, FillLayer&, const CSSValue&);
    static void mapFillComposite(CSSPropertyID, FillLayer&, const CSSValue&);
    static void mapFillBlendMode(CSSPropertyID, FillLayer&, const CSSValue&);
    static void mapFillOrigin(CSSPropertyID, FillLayer&, const CSSValue&);
    void mapFillImage(CSSPropertyID, FillLayer&, const CSSValue&);
    static void mapFillRepeat(CSSPropertyID, FillLayer&, const CSSValue&);
    void mapFillSize(CSSPropertyID, FillLayer&, const CSSValue&);
    void mapFillXPosition(CSSPropertyID, FillLayer&, const CSSValue&);
    void mapFillYPosition(CSSPropertyID, FillLayer&, const CSSValue&);
    static void mapFillMaskMode(CSSPropertyID, FillLayer&, const CSSValue&);

    void mapAnimationDelay(Animation&, const CSSValue&);
    static void mapAnimationDirection(Animation&, const CSSValue&);
    void mapAnimationDuration(Animation&, const CSSValue&);
    static void mapAnimationFillMode(Animation&, const CSSValue&);
    void mapAnimationIterationCount(Animation&, const CSSValue&);
    void mapAnimationName(Animation&, const CSSValue&);
    static void mapAnimationPlayState(Animation&, const CSSValue&);
    static void mapAnimationProperty(Animation&, const CSSValue&);
    void mapAnimationTimeline(Animation&, const CSSValue&);
    void mapAnimationTimingFunction(Animation&, const CSSValue&);
    static void mapAnimationCompositeOperation(Animation&, const CSSValue&);
    static void mapAnimationAllowsDiscreteTransitions(Animation&, const CSSValue&);
    void mapAnimationRangeStart(Animation&, const CSSValue&);
    void mapAnimationRangeEnd(Animation&, const CSSValue&);

    void mapNinePieceImage(const CSSValue*, NinePieceImage&);
    void mapNinePieceImageSlice(const CSSValue&, NinePieceImage&);
    void mapNinePieceImageSlice(const CSSBorderImageSliceValue&, NinePieceImage&);
    void mapNinePieceImageWidth(const CSSValue&, NinePieceImage&);
    void mapNinePieceImageWidth(const CSSBorderImageWidthValue&, NinePieceImage&);
    LengthBox mapNinePieceImageQuad(const CSSValue&);
    static void mapNinePieceImageRepeat(const CSSValue&, NinePieceImage&);

private:
    RefPtr<StyleImage> styleImage(const CSSValue&);
    LengthBox mapNinePieceImageQuad(const Quad&);
    Length mapNinePieceImageSide(const CSSValue&);

    Style::BuilderState& m_builderState;
};

} // namespace WebCore
