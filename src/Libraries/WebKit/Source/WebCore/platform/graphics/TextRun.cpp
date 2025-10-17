/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 28, 2022.
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
#include "TextRun.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(TextRun);

struct ExpectedTextRunSize final : public CanMakeCheckedPtr<ExpectedTextRunSize> {
    WTF_MAKE_STRUCT_FAST_ALLOCATED;
    WTF_STRUCT_OVERRIDE_DELETE_FOR_CHECKED_PTR(ExpectedTextRunSize);

    String text;
    TabSize tabSize;
    float float1;
    float float2;
    float float3;
    ExpansionBehavior expansionBehavior;
    TextSpacing::SpacingState spacingState;
    unsigned bitfields : 5;
};

static_assert(sizeof(TextRun) == sizeof(ExpectedTextRunSize), "TextRun should be small");

TextStream& operator<<(TextStream& ts, const TextRun& textRun)
{
    ts.dumpProperty("text", textRun.text());
    ts.dumpProperty("tab-size", textRun.tabSize());
    ts.dumpProperty("x-pos", textRun.xPos());
    ts.dumpProperty("horizontal-glyph-streatch", textRun.horizontalGlyphStretch());
    ts.dumpProperty("expansion", textRun.expansion());
    ts.dumpProperty("expansion-behavior", textRun.expansionBehavior());
    ts.dumpProperty("allow-tabs", textRun.allowTabs());
    ts.dumpProperty("direction", textRun.direction());
    ts.dumpProperty("directional-override", textRun.directionalOverride());
    ts.dumpProperty("character-scan-for-code-path", textRun.characterScanForCodePath());
    ts.dumpProperty("spacing-disabled", textRun.spacingDisabled());
    return ts;
}

}
