/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 8, 2023.
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

#include <optional>
#include <wtf/Forward.h>

namespace WebCore {

namespace CSS {
struct ColorScheme;
}

class CSSValue;
class CSSParserTokenRange;
struct CSSParserContext;

namespace CSSPropertyParserHelpers {

#if ENABLE(DARK_MODE_CSS)

// <'color-scheme'> = normal | [ light | dark | <custom-ident> ]+ && only?
// https://drafts.csswg.org/css-color-adjust/#propdef-color-scheme

// MARK: <'color-scheme'> consuming (unresolved)
std::optional<CSS::ColorScheme> consumeUnresolvedColorScheme(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <'color-scheme'> parsing (unresolved)
std::optional<CSS::ColorScheme> parseUnresolvedColorScheme(const String&, const CSSParserContext&);

// MARK: <'color-scheme'> consuming (CSSValue)
RefPtr<CSSValue> consumeColorScheme(CSSParserTokenRange&, const CSSParserContext&);

#endif // ENABLE(DARK_MODE_CSS)

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
