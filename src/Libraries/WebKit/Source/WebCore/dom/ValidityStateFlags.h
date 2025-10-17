/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 23, 2024.
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

namespace WebCore {

struct ValidityStateFlags {
    bool valueMissing : 1 { false };
    bool typeMismatch : 1 { false };
    bool patternMismatch : 1 { false };
    bool tooLong : 1 { false };
    bool tooShort : 1 { false };
    bool rangeUnderflow : 1 { false };
    bool rangeOverflow : 1 { false };
    bool stepMismatch : 1 { false };
    bool badInput : 1 { false };
    bool customError : 1 { false };

    bool isValid() const
    {
        return !valueMissing && !typeMismatch && !patternMismatch && !tooLong && !tooShort && !rangeUnderflow && !rangeOverflow && !stepMismatch && !badInput && !customError;
    }
};

}
