/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 14, 2023.
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

#include "CSSColorType.h"
#include "CSSParserContext.h"
#include "CSSParserFastPaths.h"
#include "CSSPlatformColorResolutionState.h"
#include <optional>
#include <wtf/OptionSet.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class Color;
class CSSParserTokenRange;
class CSSValue;
struct CSSParserContext;

namespace CSS {
struct Color;
}

namespace CSSPropertyParserHelpers {

// Options to augment color parsing.
struct CSSColorParsingOptions {
    bool acceptQuirkyColors = false;
    OptionSet<CSS::ColorType> allowedColorTypes = { CSS::ColorType::Absolute, CSS::ColorType::Current, CSS::ColorType::System };
};

// MARK: <color> consuming (unresolved)
std::optional<CSS::Color> consumeUnresolvedColor(CSSParserTokenRange&, const CSSParserContext&, const CSSColorParsingOptions& = { });

// MARK: <color> consuming (CSSValue)
RefPtr<CSSValue> consumeColor(CSSParserTokenRange&, const CSSParserContext&, const CSSColorParsingOptions& = { });

// MARK: <color> consuming (raw)
WebCore::Color consumeColorRaw(CSSParserTokenRange&, const CSSParserContext&, const CSSColorParsingOptions&, CSS::PlatformColorResolutionState&);

// MARK: <color> parsing (raw)
WEBCORE_EXPORT WebCore::Color parseColorRawSlow(const String&, const CSSParserContext&, const CSSColorParsingOptions&, CSS::PlatformColorResolutionState&);

template<typename F> WebCore::Color parseColorRaw(const String& string, const CSSParserContext& context, F&& lazySlowPathOptionsFunctor)
{
    bool strict = !isQuirksModeBehavior(context.mode);
    if (auto color = CSSParserFastPaths::parseSimpleColor(string, strict))
        return *color;

    // To avoid doing anything unnecessary before the fast path can run, callers bundle up
    // a functor to generate the slow path parameters.
    auto [options, eagerResolutionState, eagerResolutionDelegate] = lazySlowPathOptionsFunctor();

    // If a delegate is provided, hook it up to the context here. By having it live on the stack,
    // we avoid allocating it.
    if (eagerResolutionDelegate)
        eagerResolutionState.delegate = &eagerResolutionDelegate.value();

    return parseColorRawSlow(string, context, options, eagerResolutionState);
}

} // namespace CSSPropertyParserHelpers
} // namespace WebCore
