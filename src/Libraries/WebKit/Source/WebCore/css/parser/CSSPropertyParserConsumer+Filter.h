/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 11, 2024.
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
struct AppleColorFilterProperty;
struct FilterProperty;
}

class CSSParserTokenRange;
class CSSValue;
class Document;
class FilterOperations;
class RenderStyle;

struct CSSParserContext;

namespace CSSPropertyParserHelpers {

// https://drafts.fxtf.org/filter-effects/#FilterProperty

// MARK: <'filter'> consuming (CSSValue)
RefPtr<CSSValue> consumeFilter(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <'-apple-color-filter'> consuming (CSSValue)
RefPtr<CSSValue> consumeAppleColorFilter(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <'filter'> consuming (unresolved)
std::optional<CSS::FilterProperty> consumeUnresolvedFilter(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <'apple-color-filter'> consuming (unresolved)
std::optional<CSS::AppleColorFilterProperty> consumeUnresolvedAppleColorFilter(CSSParserTokenRange&, const CSSParserContext&);

// MARK: <'filter'> parsing (raw)
std::optional<FilterOperations> parseFilterValueListOrNoneRaw(const String&, const CSSParserContext&, const Document&, RenderStyle&);

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
