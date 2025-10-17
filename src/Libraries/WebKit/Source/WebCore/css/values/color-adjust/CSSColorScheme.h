/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 21, 2025.
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

#include "CSSValueTypes.h"
#include <wtf/Vector.h>
#include <wtf/text/AtomString.h>

#if ENABLE(DARK_MODE_CSS)

namespace WebCore {
namespace CSS {

// <'color-scheme'> = normal | [ light | dark | <custom-ident> ]+ && only?
// https://drafts.csswg.org/css-color-adjust/#propdef-color-scheme
struct ColorScheme {
    SpaceSeparatedVector<CustomIdentifier> schemes;
    std::optional<Keyword::Only> only;

    // As an optimization, if `schemes` is empty, that indicates the
    // entire value should be considered `normal`.
    bool isNormal() const { return schemes.isEmpty(); }

    bool operator==(const ColorScheme&) const = default;
};

template<> struct Serialize<ColorScheme> { void operator()(StringBuilder&, const ColorScheme&); };

template<size_t I> const auto& get(const ColorScheme& colorScheme)
{
    if constexpr (!I)
        return colorScheme.schemes;
    else if constexpr (I == 1)
        return colorScheme.only;
}

} // namespace CSS
} // namespace WebCore

DEFINE_TUPLE_LIKE_CONFORMANCE(WebCore::CSS::ColorScheme, 2)

#endif
