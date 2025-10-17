/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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

#include "AirOpcode.h"

namespace JSC::B3::Air {

inline Air::Opcode moveForType(Type type)
{
    switch (type.kind()) {
    case Int32:
        return Move32;
    case Int64:
        return Move;
    case Float:
        return MoveFloat;
    case Double:
        return MoveDouble;
    case V128:
        ASSERT(Options::useWasmSIMD());
        return MoveVector;
    case Void:
    case Tuple:
        break;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return Air::Oops;
}

inline Air::Opcode relaxedMoveForType(Type type)
{
    switch (type.kind()) {
    case Int32:
    case Int64:
        // For Int32, we could return Move or Move32. It's a trade-off.
        //
        // Move32: Using Move32 guarantees that we use the narrower move, but in cases where the
        //     register allocator can't prove that the variables involved are 32-bit, this will
        //     disable coalescing.
        //
        // Move: Using Move guarantees that the register allocator can coalesce normally, but in
        //     cases where it can't prove that the variables are 32-bit and it doesn't coalesce,
        //     this will force us to use a full 64-bit Move instead of the slightly cheaper
        //     32-bit Move32.
        //
        // Coalescing is a lot more profitable than turning Move into Move32. So, it's better to
        // use Move here because in cases where the register allocator cannot prove that
        // everything is 32-bit, we still get coalescing.
        return Move;
    case Float:
        // MoveFloat is always coalescable and we never convert MoveDouble to MoveFloat, so we
        // should use MoveFloat when we know that the temporaries involved are 32-bit.
        return MoveFloat;
    case Double:
        return MoveDouble;
    case V128:
        return MoveVector;
    case Void:
    case Tuple:
        break;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return Air::Oops;
}

} // namespace JSC::B3::Air

#endif // ENABLE(B3_JIT)
