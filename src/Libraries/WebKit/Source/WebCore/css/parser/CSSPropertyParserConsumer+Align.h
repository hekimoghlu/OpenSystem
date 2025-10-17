/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 31, 2022.
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

// MARK: <'align-content'>
// https://drafts.csswg.org/css-align/#propdef-align-content
RefPtr<CSSValue> consumeAlignContent(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <'justify-content'>
// https://drafts.csswg.org/css-align/#propdef-justify-content
RefPtr<CSSValue> consumeJustifyContent(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <'align-self'>
// https://drafts.csswg.org/css-align/#propdef-align-self
RefPtr<CSSValue> consumeAlignSelf(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <'justify-self'>
// https://drafts.csswg.org/css-align/#propdef-justify-self
RefPtr<CSSValue> consumeJustifySelf(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <'align-items'>
// https://drafts.csswg.org/css-align/#propdef-align-items
RefPtr<CSSValue> consumeAlignItems(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <'justify-items'>
// https://drafts.csswg.org/css-align/#propdef-justify-items
RefPtr<CSSValue> consumeJustifyItems(CSSParserTokenRange&, const CSSParserContext&);

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
