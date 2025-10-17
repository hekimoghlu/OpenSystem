/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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

#include "InlineRunAndOffset.h"
#include "RenderObjectEnums.h"

#if HAVE(REDESIGNED_TEXT_CURSOR)
#include <pal/spi/cocoa/FeatureFlagsSPI.h>
#endif

namespace WebCore {

enum class CaretRectMode {
    Normal,
    ExpandToEndOfLine
};

int caretWidth();

LayoutRect computeLocalCaretRect(const RenderObject&, const InlineBoxAndOffset&, CaretRectMode = CaretRectMode::Normal);

// FIXME: Remove this feature flag check when possible (rdar://110802729).
#if HAVE(REDESIGNED_TEXT_CURSOR)
static inline bool redesignedTextCursorEnabled()
{
    static bool enabled;
    static std::once_flag flag;
    std::call_once(flag, [] {
        enabled = os_feature_enabled(UIKit, redesigned_text_cursor);
    });
    return enabled;
}
#endif

};
