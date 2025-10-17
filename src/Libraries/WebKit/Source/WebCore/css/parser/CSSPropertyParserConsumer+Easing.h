/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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

#include "CSSEasingFunction.h"

namespace WebCore {

class CSSParserTokenRange;
class CSSToLengthConversionData;
class CSSValue;
class TimingFunction;

struct CSSParserContext;

namespace CSSPropertyParserHelpers {

// <easing-function> = linear | ease | ease-in | ease-out | ease-in-out | step-start | step-end | <linear()> | <cubic-bezier()> | <steps()>
// NOTE: also includes non-standard <spring()>.
// https://drafts.csswg.org/css-easing/#typedef-easing-function

// MARK: <easing-function> consuming (unresolved)
std::optional<CSS::EasingFunction> consumeUnresolvedEasingFunction(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <easing-function> consuming (CSSValue)
RefPtr<CSSValue> consumeEasingFunction(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <easing-function> parsing (raw)
RefPtr<TimingFunction> parseEasingFunction(const String&, const CSSParserContext&, const CSSToLengthConversionData&);
RefPtr<TimingFunction> parseEasingFunctionDeprecated(const String&, const CSSParserContext&);

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
