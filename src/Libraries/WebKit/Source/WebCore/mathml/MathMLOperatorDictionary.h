/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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

#if ENABLE(MATHML)

#include <optional>
#include <unicode/utypes.h>
#include <wtf/Forward.h>

namespace WebCore {

namespace MathMLOperatorDictionary {
enum Form { Infix, Prefix, Postfix };
enum Flag {
    Accent = 0x1,
    Fence = 0x2, // This has no visual effect but allows to expose semantic information via the accessibility tree.
    LargeOp = 0x4,
    MovableLimits = 0x8,
    Separator = 0x10, // This has no visual effect but allows to expose semantic information via the accessibility tree.
    Stretchy = 0x20,
    Symmetric = 0x40
};
const unsigned allFlags = Accent | Fence | LargeOp | MovableLimits | Separator | Stretchy | Symmetric;
struct Property {
    MathMLOperatorDictionary::Form form;
    // Default leading and trailing spaces are "thickmathspace".
    unsigned short leadingSpaceInMathUnit { 5 };
    unsigned short trailingSpaceInMathUnit { 5 };
    // Default operator properties are all set to "false".
    unsigned short flags { 0 };
};
std::optional<Property> search(char32_t, Form, bool explicitForm);
bool isVertical(char32_t);
}

}
#endif // ENABLE(MATHML)
