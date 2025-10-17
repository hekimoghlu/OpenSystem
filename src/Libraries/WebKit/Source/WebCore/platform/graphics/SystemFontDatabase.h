/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 23, 2021.
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

#include "FontSelectionAlgorithm.h"
#include <array>
#include <optional>
#include <wtf/EnumeratedArray.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class SystemFontDatabase {
public:
    WEBCORE_EXPORT static SystemFontDatabase& singleton();

    enum class FontShorthand {
        // This needs to be kept in sync with CSSValue and CSSPropertyParserHelpers::lowerFontShorthand().
        Caption,
        Icon,
        Menu,
        MessageBox,
        SmallCaption,
        WebkitMiniControl,
        WebkitSmallControl,
        WebkitControl,
#if PLATFORM(COCOA)
        AppleSystemHeadline,
        AppleSystemBody,
        AppleSystemSubheadline,
        AppleSystemFootnote,
        AppleSystemCaption1,
        AppleSystemCaption2,
        AppleSystemShortHeadline,
        AppleSystemShortBody,
        AppleSystemShortSubheadline,
        AppleSystemShortFootnote,
        AppleSystemShortCaption1,
        AppleSystemTallBody,
        AppleSystemTitle0,
        AppleSystemTitle1,
        AppleSystemTitle2,
        AppleSystemTitle3,
        AppleSystemTitle4,
#endif
        StatusBar, // This has to be kept in sync with SystemFontShorthandCache below.
    };
    using FontShorthandUnderlyingType = std::underlying_type<FontShorthand>::type;

    const AtomString& systemFontShorthandFamily(FontShorthand);
    float systemFontShorthandSize(FontShorthand);
    FontSelectionValue systemFontShorthandWeight(FontShorthand);

protected:
    SystemFontDatabase();

private:
    friend class FontCache;

    void invalidate();
    void platformInvalidate();

    struct SystemFontShorthandInfo {
        AtomString family;
        float size;
        FontSelectionValue weight;
    };
    const SystemFontShorthandInfo& systemFontShorthandInfo(FontShorthand);
    static SystemFontShorthandInfo platformSystemFontShorthandInfo(FontShorthand);

    using SystemFontShorthandCache = EnumeratedArray<FontShorthand, std::optional<SystemFontShorthandInfo>, FontShorthand::StatusBar>;
    SystemFontShorthandCache m_systemFontShorthandCache;
};

} // namespace WebCore
