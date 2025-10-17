/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 6, 2025.
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

#include "AirArg.h"
#include "AirHelpers.h"
#include "AirOpcode.h"
#include "B3AtomicValue.h"

namespace JSC { namespace B3 {

inline bool MemoryValue::isLegalOffsetImpl(int32_t offset) const
{
    // NOTE: This is inline because it constant-folds to true on x86!
    
    // So far only X86 allows exotic loads to have an offset.
    if (requiresSimpleAddr())
        return !offset;

    // The opcode is only used on ARM and Air::Move is appropriate for
    // loads/stores.
    return Air::Arg::isValidAddrForm(Air::moveForType(accessType()), offset, accessWidth());
}

inline bool MemoryValue::requiresSimpleAddr() const
{
    return !isX86() && isExotic();
}

inline Width MemoryValue::accessWidth() const
{
    switch (opcode()) {
    case Load8Z:
    case Load8S:
    case Store8:
        return Width8;
    case Load16Z:
    case Load16S:
    case Store16:
        return Width16;
    case Load:
        return widthForType(type());
    case Store:
        return widthForType(child(0)->type());
    case AtomicWeakCAS:
    case AtomicStrongCAS:
    case AtomicXchgAdd:
    case AtomicXchgAnd:
    case AtomicXchgOr:
    case AtomicXchgSub:
    case AtomicXchgXor:
    case AtomicXchg:
        return as<AtomicValue>()->accessWidth();
    default:
        RELEASE_ASSERT_NOT_REACHED();
        return Width8;
    }
}

inline bool MemoryValue::isCanonicalWidth() const
{
    return JSC::isCanonicalWidth(accessWidth());
}

} } // namespace JSC::B3

#endif // ENABLE(B3_JIT)

