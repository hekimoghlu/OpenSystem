/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 24, 2024.
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

// MARK: <'text-indent'> consuming
// https://drafts.csswg.org/css-text-3/#text-indent-property
RefPtr<CSSValue> consumeTextIndent(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <'text-transform'> consuming
// https://drafts.csswg.org/css-text-3/#text-transform-property
RefPtr<CSSValue> consumeTextTransform(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <'hanging-punctuation'> consuming
// https://drafts.csswg.org/css-text-3/#propdef-hanging-punctuation
RefPtr<CSSValue> consumeHangingPunctuation(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <'text-autospace'> consuming
// https://drafts.csswg.org/css-text-4/#text-autospace-property
RefPtr<CSSValue> consumeTextAutospace(CSSParserTokenRange&, const CSSParserContext&);

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
