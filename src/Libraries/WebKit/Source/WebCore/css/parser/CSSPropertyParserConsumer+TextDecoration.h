/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 17, 2024.
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

// MARK: <'text-shadow'> consuming
// https://drafts.csswg.org/css-text-decor-3/#propdef-text-shadow
RefPtr<CSSValue> consumeTextShadow(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <'text-decoration-line'> consuming
// https://drafts.csswg.org/css-text-decor-3/#text-decoration-line-property
RefPtr<CSSValue> consumeTextDecorationLine(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <'text-emphasis-style'> consuming
// https://drafts.csswg.org/css-text-decor-3/#text-emphasis-style-property
RefPtr<CSSValue> consumeTextEmphasisStyle(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <'text-emphasis-position'> consuming
// https://drafts.csswg.org/css-text-decor-3/#text-emphasis-position-property
RefPtr<CSSValue> consumeTextEmphasisPosition(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <'text-underline-position'> consuming
// https://drafts.csswg.org/css-text-decor-4/#text-underline-position-property
RefPtr<CSSValue> consumeTextUnderlinePosition(CSSParserTokenRange&, const CSSParserContext&);

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
