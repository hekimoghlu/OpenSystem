/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 7, 2023.
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

class CSSParserTokenRange;
class CSSValue;
struct CSSParserContext;

namespace CSSPropertyParserHelpers {

enum class PathParsingOption : uint8_t {
    None = 0,
    RejectPathFillRule = 1 << 0,
    RejectPath = 1 << 1,
};

// <basic-shape> = <circle()> | <ellipse() | <inset()> | <path()> | <polygon()> | <rect()> | <shape()> | <xywh()>
// https://drafts.csswg.org/css-shapes/#typedef-basic-shape
RefPtr<CSSValue> consumeBasicShape(CSSParserTokenRange&, const CSSParserContext&, OptionSet<PathParsingOption>);

// <path()> = path( <'fill-rule'>? , <string> )
// https://drafts.csswg.org/css-shapes/#funcdef-basic-shape-path
RefPtr<CSSValue> consumePath(CSSParserTokenRange&, const CSSParserContext&);

// <'shape-outside'> = none | [ <basic-shape> || <shape-box> ] | <image>
// https://drafts.csswg.org/css-shapes/#propdef-shape-outside
RefPtr<CSSValue> consumeShapeOutside(CSSParserTokenRange&, const CSSParserContext&);


} // namespace CSSPropertyParserHelpers
} // namespace WebCore
