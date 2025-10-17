/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 18, 2023.
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
#include "OpenTypeCG.h"

#include "OpenTypeTypes.h"
#include <wtf/StdLibExtras.h>

#if USE(CF)
#include <wtf/cf/VectorCF.h>
#endif

namespace WebCore {
namespace OpenType {

static inline short readShortFromTable(std::span<const UInt8> os2Data, CFIndex offset)
{
    return reinterpretCastSpanStartTo<const OpenType::Int16>(os2Data.subspan(offset));
}

bool tryGetTypoMetrics(CTFontRef font, short& ascent, short& descent, short& lineGap)
{
    bool result = false;
    if (auto os2Table = adoptCF(CTFontCopyTable(font, kCTFontTableOS2, kCTFontTableOptionNoOptions))) {
        // For the structure of the OS/2 table, see
        // https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6OS2.html
        const CFIndex fsSelectionOffset = 16 * 2 + 10 + 4 * 4 + 4 * 1;
        const CFIndex sTypoAscenderOffset = fsSelectionOffset + 3 * 2;
        const CFIndex sTypoDescenderOffset = sTypoAscenderOffset + 2;
        const CFIndex sTypoLineGapOffset = sTypoDescenderOffset + 2;
        if (CFDataGetLength(os2Table.get()) >= sTypoLineGapOffset + 2) {
            auto os2Data = span(os2Table.get());
            // We test the use typo bit on the least significant byte of fsSelection.
            const UInt8 useTypoMetricsMask = 1 << 7;
            if (os2Data[fsSelectionOffset + 1] & useTypoMetricsMask) {
                ascent = readShortFromTable(os2Data, sTypoAscenderOffset);
                descent = readShortFromTable(os2Data, sTypoDescenderOffset);
                lineGap = readShortFromTable(os2Data, sTypoLineGapOffset);
                result = true;
            }
        }
    }
    return result;
}

} // namespace OpenType
} // namespace WebCore
