/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 24, 2023.
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

#include "ArgumentCoders.h"
#include <wtf/RefPtr.h>

namespace WebCore {
class LocalFrame;
struct SimpleRange;
}

namespace WebKit {

enum class EditingRangeIsRelativeTo : uint8_t {
    EditableRoot,
    Paragraph,
};

// FIXME: Replace this with std::optional<CharacterRange>.
struct EditingRange {
    EditingRange()
        : location(notFound)
        , length(0)
    {
    }

    EditingRange(uint64_t location, uint64_t length)
        : location(location)
        , length(length)
    {
    }

    // (notFound, 0) is notably valid.
    bool isValid() const { return location + length >= location; }

    static std::optional<WebCore::SimpleRange> toRange(WebCore::LocalFrame&, const EditingRange&, EditingRangeIsRelativeTo = EditingRangeIsRelativeTo::EditableRoot);
    static EditingRange fromRange(WebCore::LocalFrame&, const std::optional<WebCore::SimpleRange>&, EditingRangeIsRelativeTo = EditingRangeIsRelativeTo::EditableRoot);

#if defined(__OBJC__)
    EditingRange(NSRange range)
    {
        if (range.location != NSNotFound) {
            location = range.location;
            length = range.length;
        } else {
            location = notFound;
            length = 0;
        }
    }

    operator NSRange() const
    {
        if (location == notFound)
            return NSMakeRange(NSNotFound, 0);
        return NSMakeRange(location, length);
    }
#endif

    uint64_t location;
    uint64_t length;
};

}
