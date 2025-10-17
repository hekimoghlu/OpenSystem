/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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

#include "LayoutUnit.h"
#include <optional>

namespace WebCore {

namespace Layout {

class ElementBox;
class LayoutState;
}

namespace LayoutIntegration {

void layoutWithFormattingContextForBox(const Layout::ElementBox&, std::optional<LayoutUnit> widthConstraint, Layout::LayoutState&);

enum class LogicalWidthType : uint8_t  {
    PreferredMaximum,
    PreferredMinimum,
    MaxContent,
    MinContent
};
LayoutUnit formattingContextRootLogicalWidthForType(const Layout::ElementBox&, LogicalWidthType);

enum class LogicalHeightType : uint8_t  {
    MinContent
};
LayoutUnit formattingContextRootLogicalHeightForType(const Layout::ElementBox&, LogicalHeightType);

}
}
