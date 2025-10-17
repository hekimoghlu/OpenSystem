/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 7, 2022.
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

#if ENABLE(DFG_JIT)

#include "DFGNodeFlags.h"
#include "DFGUseKind.h"
#include "DataFormat.h"
#include "DumpContext.h"
#include <wtf/PrintStream.h>

namespace JSC { namespace DFG {

enum FlushFormat : uint8_t {
    DeadFlush,
    FlushedInt32,
    FlushedInt52,
    FlushedDouble,
    FlushedCell,
    FlushedBoolean,
    FlushedJSValue,
    ConflictingFlush
};

inline NodeFlags resultFor(FlushFormat format)
{
    switch (format) {
    case DeadFlush:
    case FlushedJSValue:
    case FlushedCell:
    case ConflictingFlush:
        return NodeResultJS;
    case FlushedInt32:
        return NodeResultInt32;
    case FlushedInt52:
        return NodeResultInt52;
    case FlushedDouble:
        return NodeResultDouble;
    case FlushedBoolean:
        return NodeResultBoolean;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return 0;
}

inline UseKind useKindFor(FlushFormat format)
{
    switch (format) {
    case DeadFlush:
    case FlushedJSValue:
    case ConflictingFlush:
        return UntypedUse;
    case FlushedCell:
        return CellUse;
    case FlushedInt32:
        return Int32Use;
    case FlushedInt52:
        return Int52RepUse;
    case FlushedDouble:
        return DoubleRepUse;
    case FlushedBoolean:
        return BooleanUse;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return UntypedUse;
}

inline UseKind uncheckedUseKindFor(FlushFormat format)
{
    switch (format) {
    case DeadFlush:
    case FlushedJSValue:
    case ConflictingFlush:
        return UntypedUse;
    case FlushedCell:
        return KnownCellUse;
    case FlushedInt32:
        return KnownInt32Use;
    case FlushedInt52:
        return Int52RepUse;
    case FlushedDouble:
        return DoubleRepUse;
    case FlushedBoolean:
        return KnownBooleanUse;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return UntypedUse;
}

inline SpeculatedType typeFilterFor(FlushFormat format)
{
    return typeFilterFor(useKindFor(format));
}

inline DataFormat dataFormatFor(FlushFormat format)
{
    switch (format) {
    case DeadFlush:
    case ConflictingFlush:
        return DataFormatDead;
    case FlushedJSValue:
        return DataFormatJS;
    case FlushedDouble:
        return DataFormatDouble;
    case FlushedInt32:
        return DataFormatInt32;
    case FlushedInt52:
        return DataFormatInt52;
    case FlushedCell:
        return DataFormatCell;
    case FlushedBoolean:
        return DataFormatBoolean;
    }
    RELEASE_ASSERT_NOT_REACHED();
    return DataFormatDead;
}

inline FlushFormat merge(FlushFormat a, FlushFormat b)
{
    if (a == DeadFlush)
        return b;
    if (b == DeadFlush)
        return a;
    if (a == b)
        return a;
    return ConflictingFlush;
}

inline bool isConcrete(FlushFormat format)
{
    return format != DeadFlush && format != ConflictingFlush;
}

} } // namespace JSC::DFG

namespace WTF {

void printInternal(PrintStream&, JSC::DFG::FlushFormat);

inline JSC::DFG::FlushFormat inContext(JSC::DFG::FlushFormat format, JSC::DumpContext*)
{
    return format;
}

} // namespace WTF

#endif // ENABLE(DFG_JIT)
