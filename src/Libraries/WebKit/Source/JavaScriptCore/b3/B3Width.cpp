/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 14, 2022.
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
#include "B3Width.h"

#if ENABLE(B3_JIT)

#include <wtf/PrintStream.h>

namespace JSC { namespace B3 {

Type bestType(Bank bank, Width width)
{
    switch (width) {
    case Width8:
    case Width16:
    case Width32:
        switch (bank) {
        case GP:
            return Int32;
        case FP:
            return Float;
        }
        RELEASE_ASSERT_NOT_REACHED();
        return Void;
    case Width64:
        switch (bank) {
        case GP:
            return Int64;
        case FP:
            return Double;
        }
        RELEASE_ASSERT_NOT_REACHED();
        return Void;
    case Width128:
        break;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return Void;
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

