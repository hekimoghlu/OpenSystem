/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 24, 2025.
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

#if ENABLE(B3_JIT)

#include "B3Bank.h"
#include "B3Type.h"
#include "SIMDInfo.h"
#include "Width.h"

#if !ASSERT_ENABLED
IGNORE_RETURN_TYPE_WARNINGS_BEGIN
#endif

namespace JSC { namespace B3 {

inline Width widthForType(Type type)
{
    switch (type.kind()) {
    case Void:
    case Tuple:
        ASSERT_NOT_REACHED();
        return Width8;
    case Int32:
    case Float:
        return Width32;
    case Int64:
    case Double:
        return Width64;
    case V128:
        return Width128;
    }
    ASSERT_NOT_REACHED();
    return Width8;
}

Type bestType(Bank bank, Width width);

} } // namespace JSC::B3

#if !ASSERT_ENABLED
IGNORE_RETURN_TYPE_WARNINGS_END
#endif

#endif // ENABLE(B3_JIT)

