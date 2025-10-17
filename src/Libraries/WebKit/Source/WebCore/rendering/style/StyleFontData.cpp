/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 20, 2024.
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
#include "StyleFontData.h"

namespace WebCore {

StyleFontData::StyleFontData() = default;

StyleFontData::StyleFontData(const StyleFontData& o)
    : fontCascade(o.fontCascade)
{
}

Ref<StyleFontData> StyleFontData::copy() const
{
    return adoptRef(*new StyleFontData(*this));
}

bool StyleFontData::operator==(const StyleFontData& o) const
{
    return fontCascade == o.fontCascade;
}

#if !LOG_DISABLED
void StyleFontData::dumpDifferences(TextStream& ts, const StyleFontData& other) const
{
    if (fontCascade != other.fontCascade)
        ts << "fontCascade differs:\n  " << fontCascade << "\n  " << other.fontCascade;
}
#endif

} // namespace WebCore
