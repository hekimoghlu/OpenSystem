/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 13, 2025.
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

#include "InlineDisplayBoxInlines.h"
#include "InlineIteratorBoxModernPath.h"

namespace WebCore {
namespace InlineIterator {

inline bool BoxModernPath::isHorizontal() const { return box().isHorizontal(); }

inline WritingMode BoxModernPath::writingMode() const { return box().writingMode(); }

inline TextRun BoxModernPath::textRun(TextRunMode mode) const
{
    auto& style = box().style();
    auto expansion = box().expansion();
    auto logicalLeft = [&] {
        if (style.writingMode().isBidiLTR())
            return visualRectIgnoringBlockDirection().x() - (line().lineBoxLeft() + line().contentLogicalLeft());
        return line().lineBoxRight() - (visualRectIgnoringBlockDirection().maxX() + line().contentLogicalLeft());
    };
    auto characterScanForCodePath = isText() && !renderText().canUseSimpleFontCodePath();
    auto textRun = TextRun { mode == TextRunMode::Editing ? originalText() : box().text().renderedContent(), logicalLeft(), expansion.horizontalExpansion, expansion.behavior, direction(), style.rtlOrdering() == Order::Visual, characterScanForCodePath };
    textRun.setTabSize(!style.collapseWhiteSpace(), style.tabSize());
    return textRun;
}

}
}
