/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 9, 2024.
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
#include "SystemFontDatabase.h"

namespace WebCore {

SystemFontDatabase::SystemFontDatabase() = default;

auto SystemFontDatabase::systemFontShorthandInfo(FontShorthand fontShorthand) -> const SystemFontShorthandInfo& {
    if (auto& entry = m_systemFontShorthandCache[fontShorthand])
        return *entry;

    m_systemFontShorthandCache[fontShorthand] = platformSystemFontShorthandInfo(fontShorthand);
    return *m_systemFontShorthandCache[fontShorthand];
}

const AtomString& SystemFontDatabase::systemFontShorthandFamily(FontShorthand fontShorthand)
{
    return systemFontShorthandInfo(fontShorthand).family;
}

float SystemFontDatabase::systemFontShorthandSize(FontShorthand fontShorthand)
{
    return systemFontShorthandInfo(fontShorthand).size;
}

FontSelectionValue SystemFontDatabase::systemFontShorthandWeight(FontShorthand fontShorthand)
{
    return systemFontShorthandInfo(fontShorthand).weight;
}

void SystemFontDatabase::invalidate()
{
    for (auto& item : m_systemFontShorthandCache)
        item.reset();
    platformInvalidate();
}

} // namespace WebCore
