/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 19, 2022.
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

#include "WritingMode.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

// In some circumstances we want to store a String along with the TextDirection
// of the String as learned from the context of the String. For example,
// consider storing the title derived from <title dir='rtl'>some title</title>
// in the history.
//
// Note that is explicitly *not* the direction of the string as learned
// from the characters of the string; it's extra metadata we have external
// to the string.

struct StringWithDirection {
    String string;
    TextDirection direction { TextDirection::LTR };

    friend bool operator==(const StringWithDirection&, const StringWithDirection&) = default;
};

inline StringWithDirection truncateFromEnd(const StringWithDirection& string, unsigned maxLength)
{
    return { string.string.left(maxLength), string.direction };
}

}
