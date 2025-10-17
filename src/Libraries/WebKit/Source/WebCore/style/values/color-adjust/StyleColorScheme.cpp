/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 29, 2022.
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
#include "config.h"
#include "StyleColorScheme.h"

#if ENABLE(DARK_MODE_CSS)

#include "CSSToLengthConversionData.h"
#include "CSSValueKeywords.h"
#include <wtf/text/TextStream.h>

namespace WebCore {
namespace Style {

OptionSet<WebCore::ColorScheme> ColorScheme::colorScheme() const
{
    OptionSet<WebCore::ColorScheme> result;
    for (auto& scheme : schemes) {
        if (equalLettersIgnoringASCIICase(scheme.value, "light"_s))
            result.add(WebCore::ColorScheme::Light);
        else if (equalLettersIgnoringASCIICase(scheme.value, "dark"_s))
            result.add(WebCore::ColorScheme::Dark);
    }
    return result;
}

WTF::TextStream& operator<<(WTF::TextStream& ts, const ColorScheme& value)
{
    if (value.isNormal())
        return ts << "normal";

    ts << value.schemes.value;
    if (value.only)
        ts << " only";

    return ts;
}

} // namespace Style
} // namespace WebCore

#endif
