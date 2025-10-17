/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 31, 2022.
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
#include "StyleScrollSnapPoints.h"

namespace WebCore {

WTF::TextStream& operator<<(WTF::TextStream& ts, ScrollSnapAlign align)
{
    auto populateTextStreamForAlignType = [&](ScrollSnapAxisAlignType type) {
        switch (type) {
        case ScrollSnapAxisAlignType::None: ts << "none"; break;
        case ScrollSnapAxisAlignType::Start: ts << "start"; break;
        case ScrollSnapAxisAlignType::Center: ts << "center"; break;
        case ScrollSnapAxisAlignType::End: ts << "end"; break;
        }
    };
    populateTextStreamForAlignType(align.blockAlign);
    if (align.blockAlign != align.inlineAlign) {
        ts << ' ';
        populateTextStreamForAlignType(align.inlineAlign);
    }
    return ts;
}

WTF::TextStream& operator<<(WTF::TextStream& ts, ScrollSnapType type)
{
    if (type.strictness != ScrollSnapStrictness::None) {
        switch (type.axis) {
        case ScrollSnapAxis::XAxis: ts << "x"; break;
        case ScrollSnapAxis::YAxis: ts << "y"; break;
        case ScrollSnapAxis::Block: ts << "block"; break;
        case ScrollSnapAxis::Inline: ts << "inline"; break;
        case ScrollSnapAxis::Both: ts << "both"; break;
        }
        ts << ' ';
    }
    switch (type.strictness) {
    case ScrollSnapStrictness::None: ts << "none"; break;
    case ScrollSnapStrictness::Proximity: ts << "proximity"; break;
    case ScrollSnapStrictness::Mandatory: ts << "mandatory"; break;
    }
    return ts;
}

} // namespace WebCore
