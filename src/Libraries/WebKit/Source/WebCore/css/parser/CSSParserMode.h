/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 2, 2022.
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

namespace WebCore {

// Must not grow beyond 3 bits, due to packing in StyleProperties.
enum CSSParserMode : uint8_t {
    HTMLStandardMode,
    HTMLQuirksMode,
    // SVG attributes are parsed in quirks mode but rules differ slightly.
    SVGAttributeMode,
    // User agent stylesheets are parsed in standards mode but also allows internal properties and values.
    UASheetMode,
    // WebVTT places limitations on external resources.
    WebVTTMode
};

inline bool isQuirksModeBehavior(CSSParserMode mode)
{
    return mode == HTMLQuirksMode;
}

inline bool isUASheetBehavior(CSSParserMode mode)
{
    return mode == UASheetMode;
}

inline bool isUnitlessValueParsingEnabledForMode(CSSParserMode mode)
{
    return mode == SVGAttributeMode;
}

// FIXME-NEWPARSER: Next two functions should be removed eventually.
inline CSSParserMode strictToCSSParserMode(bool inStrictMode)
{
    return inStrictMode ? HTMLStandardMode : HTMLQuirksMode;
}

inline bool isStrictParserMode(CSSParserMode cssParserMode)
{
    switch (cssParserMode) {
    case UASheetMode:
    case HTMLStandardMode:
    case SVGAttributeMode:
    case WebVTTMode:
        return true;
    case HTMLQuirksMode:
        return false;
    }
    ASSERT_NOT_REACHED();
    return false;
}

} // namespace WebCore
