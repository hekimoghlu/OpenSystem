/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 17, 2025.
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
#include "UnicodeHelpers.h"

#include "WritingMode.h"
#include <unicode/ubidi.h>

namespace WebCore {

std::optional<TextDirection> baseTextDirection(StringView content)
{
    auto characters = content.span16();
    switch (ubidi_getBaseDirection(characters.data(), characters.size())) {
    case UBIDI_RTL:
        return TextDirection::RTL;
    case UBIDI_LTR:
        return TextDirection::LTR;
    case UBIDI_NEUTRAL:
    case UBIDI_MIXED:
        return std::nullopt;
    }
    ASSERT_NOT_REACHED();
    return std::nullopt;
}

} // namespace WebCore
