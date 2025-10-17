/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 1, 2022.
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

#include <array>
#include <optional>

namespace WebCore {

namespace CSS {
struct BorderRadius;
}

class CSSParserTokenRange;
class CSSValue;
enum CSSPropertyID : uint16_t;
struct CSSParserContext;

namespace CSSPropertyParserHelpers {

// MARK: - Border Radius

// <'border-radius'> = <length-percentage [0,âˆž]>{1,4} [ / <length-percentage [0,âˆž]>{1,4} ]?
// https://drafts.csswg.org/css-backgrounds/#propdef-border-radius
std::optional<CSS::BorderRadius> consumeUnresolvedBorderRadius(CSSParserTokenRange&, const CSSParserContext&);

// Non-standard -webkit-border-radius.
std::optional<CSS::BorderRadius> consumeUnresolvedWebKitBorderRadius(CSSParserTokenRange&, const CSSParserContext&);

// <'border-[top|bottom]-[left|right]-radius,'> = <length-percentage [0,âˆž]>{1,2}
// https://drafts.csswg.org/css-backgrounds/#propdef-border-top-left-radius
RefPtr<CSSValue> consumeBorderRadiusCorner(CSSParserTokenRange&, const CSSParserContext&);

// MARK: - Border Image

// <'border-image-repeat> = [ stretch | repeat | round | space ]{1,2}
// https://drafts.csswg.org/css-backgrounds/#propdef-border-image-repeat
RefPtr<CSSValue> consumeBorderImageRepeat(CSSParserTokenRange&, const CSSParserContext&);

// <'border-image-slice'> = [<number [0,âˆž]> | <percentage [0,âˆž]>]{1,4} && fill?
// https://drafts.csswg.org/css-backgrounds/#propdef-border-image-slice
RefPtr<CSSValue> consumeBorderImageSlice(CSSParserTokenRange&, const CSSParserContext&, CSSPropertyID currentProperty);

// <'border-image-outset'> = [ <length [0,âˆž]> | <number [0,âˆž]> ]{1,4}
// https://drafts.csswg.org/css-backgrounds/#propdef-border-image-outset
RefPtr<CSSValue> consumeBorderImageOutset(CSSParserTokenRange&, const CSSParserContext&);

// <'border-image-width'> = [ <length-percentage [0,âˆž]> | <number [0,âˆž]> | auto ]{1,4}
// https://drafts.csswg.org/css-backgrounds/#propdef-border-image-width
RefPtr<CSSValue> consumeBorderImageWidth(CSSParserTokenRange&, const CSSParserContext&, CSSPropertyID currentProperty);

// https://drafts.csswg.org/css-backgrounds/#border-image
bool consumeBorderImageComponents(CSSParserTokenRange&, const CSSParserContext&, CSSPropertyID currentProperty, RefPtr<CSSValue>&, RefPtr<CSSValue>&, RefPtr<CSSValue>&, RefPtr<CSSValue>&, RefPtr<CSSValue>&);

// MARK: - Border Style

// <'border-*-width'> = <line-width>
// https://drafts.csswg.org/css-backgrounds/#propdef-border-top-width
RefPtr<CSSValue> consumeBorderWidth(CSSParserTokenRange&, const CSSParserContext&, CSSPropertyID currentShorthand);

// <'border-*-width'> = <line-width>
// https://drafts.csswg.org/css-backgrounds/#propdef-border-top-width
RefPtr<CSSValue> consumeBorderColor(CSSParserTokenRange&, const CSSParserContext&, CSSPropertyID currentShorthand);

// MARK: - Background Clip

// <single-background-clip> = <visual-box>
// https://drafts.csswg.org/css-backgrounds/#propdef-background-clip
RefPtr<CSSValue> consumeSingleBackgroundClip(CSSParserTokenRange&, const CSSParserContext&);

// <'background-clip'> = <visual-box>#
// https://drafts.csswg.org/css-backgrounds/#propdef-background-clip
RefPtr<CSSValue> consumeBackgroundClip(CSSParserTokenRange&, const CSSParserContext&);

// MARK: - Background Size

// <bg-size> = [ <length-percentage [0,âˆž]> | auto ]{1,2} | cover | contain
// https://drafts.csswg.org/css-backgrounds/#background-size
RefPtr<CSSValue> consumeSingleBackgroundSize(CSSParserTokenRange&, const CSSParserContext&);

// Non-standard.
RefPtr<CSSValue> consumeSingleWebkitBackgroundSize(CSSParserTokenRange&, const CSSParserContext&);

// <single-mask-size> = <bg-size>
// https://drafts.fxtf.org/css-masking/#the-mask-size
RefPtr<CSSValue> consumeSingleMaskSize(CSSParserTokenRange&, const CSSParserContext&);

// MARK: - Background Repeat

// <repeat-style> = repeat-x | repeat-y | [repeat | space | round | no-repeat]{1,2}
// https://drafts.csswg.org/css-backgrounds/#typedef-repeat-style
RefPtr<CSSValue> consumeRepeatStyle(CSSParserTokenRange&, const CSSParserContext&);

// MARK: - Shadows

// <'box-shadow'> = none | <shadow>#
// https://drafts.csswg.org/css-backgrounds/#propdef-box-shadow
RefPtr<CSSValue> consumeBoxShadow(CSSParserTokenRange&, const CSSParserContext&);
RefPtr<CSSValue> consumeWebkitBoxShadow(CSSParserTokenRange&, const CSSParserContext&);

// MARK: - Reflection (non-standard)

// Non-standard addition.
RefPtr<CSSValue> consumeReflect(CSSParserTokenRange&, const CSSParserContext&);

// MARK: Utilities for filling in rects / quads in the "margin" form.

// - if only 1 value, `a`, is provided, set top, bottom, right & left to `a`.
// - if only 2 values, `a` and `b` are provided, set top & bottom to `a`, right & left to `b`.
// - if only 3 values, `a`, `b`, and `c` are provided, set top to `a`, right to `b`, bottom to `c`, & left to `b`.

template<typename Container, typename T> Container completeQuad(T a)
{
    return Container { a, a, a, a };
}

template<typename Container, typename T> Container completeQuad(T a, T b)
{
    return Container { a, b, a, b };
}

template<typename Container, typename T> Container completeQuad(T a, T b, T c)
{
    return Container { a, b, c, b };
}

template<typename Container, typename T> Container completeQuadFromArray(std::array<std::optional<T>, 4> optionals)
{
    ASSERT(optionals[0].has_value());

    if (!optionals[1])
        return completeQuad<Container>(WTFMove(*optionals[0]));

    if (!optionals[2])
        return completeQuad<Container>(WTFMove(*optionals[0]), WTFMove(*optionals[1]));

    if (!optionals[3])
        return completeQuad<Container>(WTFMove(*optionals[0]), WTFMove(*optionals[1]), WTFMove(*optionals[2]));

    return Container { WTFMove(*optionals[0]), WTFMove(*optionals[1]), WTFMove(*optionals[2]), WTFMove(*optionals[3]) };
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
