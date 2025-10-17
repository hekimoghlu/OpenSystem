/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 4, 2024.
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
#include "SpeculatedType.h"
#include <wtf/PrintStream.h>

namespace JSC { namespace DFG {

enum UseKind : uint8_t {
    // The DFG has 3 representations of values used:

    // 1. The JSValue representation for a JSValue that must be stored in a GP
    //    register (or a GP register pair), and follows rules for boxing and unboxing
    //    that allow the JSValue to be stored as either fully boxed JSValues, or
    //    unboxed Int32, Booleans, Cells, etc. in 32-bit as appropriate.
    UntypedUse, // UntypedUse must come first (value 0).
    Int32Use,
    KnownInt32Use,
    AnyIntUse,
    NumberUse,
    RealNumberUse,
    BooleanUse,
    KnownBooleanUse,
    CellUse,
    KnownCellUse,
    CellOrOtherUse,
    ObjectUse,
    ArrayUse,
    FunctionUse,
    FinalObjectUse,
    PromiseObjectUse,
    RegExpObjectUse,
    ProxyObjectUse,
    GlobalProxyUse,
    DerivedArrayUse,
    ObjectOrOtherUse,
    StringIdentUse,
    StringUse,
    StringOrOtherUse,
    KnownStringUse,
    KnownPrimitiveUse, // This bizarre type arises for op_strcat, which has a bytecode guarantee that it will only see primitives (i.e. not objects).
    SymbolUse,
    AnyBigIntUse,
    HeapBigIntUse,
    BigInt32Use,
    DateObjectUse,
    MapObjectUse,
    SetObjectUse,
    MapIteratorObjectUse,
    SetIteratorObjectUse,
    WeakMapObjectUse,
    WeakSetObjectUse,
    DataViewObjectUse,
    StringObjectUse,
    StringOrStringObjectUse,
    NotStringVarUse,
    NotSymbolUse,
    NotCellUse,
    NotCellNorBigIntUse,
    NotDoubleUse,
    NeitherDoubleNorHeapBigIntUse,
    NeitherDoubleNorHeapBigIntNorStringUse,
    KnownOtherUse,
    OtherUse,
    MiscUse,

    // 2. The Double representation for an unboxed double value that must be stored
    //    in an FP register.
    DoubleRepUse,
    DoubleRepRealUse,
    DoubleRepAnyIntUse,

    // 3. The Int52 representation for an unboxed integer value that must be stored
    //    in a GP register.
    Int52RepUse,

    LastUseKind // Must always be the last entry in the enum, as it is used to denote the number of enum elements.
};

inline SpeculatedType typeFilterFor(UseKind useKind)
{
    switch (useKind) {
    case UntypedUse:
        return SpecBytecodeTop;
    case Int32Use:
    case KnownInt32Use:
        return SpecInt32Only;
    case Int52RepUse:
        return SpecInt52Any;
    case AnyIntUse:
        return SpecInt32Only | SpecAnyIntAsDouble;
    case NumberUse:
        return SpecBytecodeNumber;
    case RealNumberUse:
        return SpecBytecodeRealNumber;
    case DoubleRepUse:
        return SpecFullDouble;
    case DoubleRepRealUse:
        return SpecDoubleReal;
    case DoubleRepAnyIntUse:
        return SpecAnyIntAsDouble;
    case BooleanUse:
    case KnownBooleanUse:
        return SpecBoolean;
    case CellUse:
    case KnownCellUse:
        return SpecCellCheck;
    case CellOrOtherUse:
        return SpecCellCheck | SpecOther;
    case ObjectUse:
        return SpecObject;
    case ArrayUse:
        return SpecArray;
    case FunctionUse:
        return SpecFunction;
    case FinalObjectUse:
        return SpecFinalObject;
    case RegExpObjectUse:
        return SpecRegExpObject;
    case ProxyObjectUse:
        return SpecProxyObject;
    case GlobalProxyUse:
        return SpecGlobalProxy;
    case DerivedArrayUse:
        return SpecDerivedArray;
    case ObjectOrOtherUse:
        return SpecObject | SpecOther;
    case StringIdentUse:
        return SpecStringIdent;
    case StringUse:
    case KnownStringUse:
        return SpecString;
    case StringOrOtherUse:
        return SpecString | SpecOther;
    case KnownPrimitiveUse:
        return SpecHeapTop & ~SpecObject;
    case SymbolUse:
        return SpecSymbol;
    case AnyBigIntUse:
        return SpecBigInt;
    case HeapBigIntUse:
        return SpecHeapBigInt;
    case BigInt32Use:
        return SpecBigInt32;
    case PromiseObjectUse:
        return SpecPromiseObject;
    case DateObjectUse:
        return SpecDateObject;
    case MapObjectUse:
        return SpecMapObject;
    case SetObjectUse:
        return SpecSetObject;
    case MapIteratorObjectUse:
    case SetIteratorObjectUse:
        return SpecObjectOther;
    case WeakMapObjectUse:
        return SpecWeakMapObject;
    case WeakSetObjectUse:
        return SpecWeakSetObject;
    case DataViewObjectUse:
        return SpecDataViewObject;
    case StringObjectUse:
        return SpecStringObject;
    case StringOrStringObjectUse:
        return SpecString | SpecStringObject;
    case NotStringVarUse:
        return ~SpecStringVar;
    case NotSymbolUse:
        return ~SpecSymbol;
    case NotCellUse:
        return ~SpecCellCheck;
    case NotCellNorBigIntUse:
        return ~SpecCellCheck & ~SpecBigInt;
    case NotDoubleUse:
        return ~SpecFullDouble;
    case NeitherDoubleNorHeapBigIntUse:
        return ~SpecFullDouble & ~SpecHeapBigInt;
    case NeitherDoubleNorHeapBigIntNorStringUse:
        return ~(SpecFullDouble | SpecHeapBigInt | SpecString);
    case KnownOtherUse:
    case OtherUse:
        return SpecOther;
    case MiscUse:
        return SpecMisc;
    default:
        RELEASE_ASSERT_NOT_REACHED();
        return SpecFullTop;
    }
}

inline bool shouldNotHaveTypeCheck(UseKind kind)
{
    switch (kind) {
    case UntypedUse:
    case KnownInt32Use:
    case KnownCellUse:
    case KnownStringUse:
    case KnownPrimitiveUse:
    case KnownBooleanUse:
    case KnownOtherUse:
    case Int52RepUse:
    case DoubleRepUse:
        return true;
    default:
        return false;
    }
}

inline bool mayHaveTypeCheck(UseKind kind)
{
    return !shouldNotHaveTypeCheck(kind);
}

inline bool isDouble(UseKind kind)
{
    switch (kind) {
    case DoubleRepUse:
    case DoubleRepRealUse:
    case DoubleRepAnyIntUse:
        return true;
    default:
        return false;
    }
}

// Returns true if the use kind only admits cells, and is therefore appropriate for
// SpeculateCellOperand in the DFG or lowCell() in the FTL.
inline bool isCell(UseKind kind)
{
    switch (kind) {
    case CellUse:
    case KnownCellUse:
    case ObjectUse:
    case ArrayUse:
    case FunctionUse:
    case FinalObjectUse:
    case RegExpObjectUse:
    case PromiseObjectUse:
    case ProxyObjectUse:
    case GlobalProxyUse:
    case DerivedArrayUse:
    case StringIdentUse:
    case StringUse:
    case KnownStringUse:
    case SymbolUse:
    case HeapBigIntUse:
    case StringObjectUse:
    case StringOrStringObjectUse:
    case DateObjectUse:
    case MapObjectUse:
    case SetObjectUse:
    case MapIteratorObjectUse:
    case SetIteratorObjectUse:
    case WeakMapObjectUse:
    case WeakSetObjectUse:
    case DataViewObjectUse:
        return true;
    default:
        return false;
    }
}

// Returns true if we've already guaranteed the type 
inline bool alreadyChecked(UseKind kind, SpeculatedType type)
{
    return !(type & ~typeFilterFor(kind));
}

inline UseKind useKindForResult(NodeFlags result)
{
    ASSERT(!(result & ~NodeResultMask));
    switch (result) {
    case NodeResultInt52:
        return Int52RepUse;
    case NodeResultDouble:
        return DoubleRepUse;
    default:
        return UntypedUse;
    }
}

inline bool checkMayCrashIfInputIsEmpty(UseKind kind)
{
#if USE(JSVALUE64)
    switch (kind) {
    case UntypedUse:
    case Int32Use:
    case KnownInt32Use:
    case AnyIntUse:
    case NumberUse:
    case BooleanUse:
    case KnownBooleanUse:
    case CellUse:
    case KnownCellUse:
    case CellOrOtherUse:
    case KnownOtherUse:
    case OtherUse:
    case MiscUse:
    case NotCellUse:
    case NotCellNorBigIntUse:
    case NotDoubleUse:
    case NeitherDoubleNorHeapBigIntUse:
    case NeitherDoubleNorHeapBigIntNorStringUse:
        return false;
    default:
        return true;
    }
#else
    UNUSED_PARAM(kind);
    return true;
#endif
}

} } // namespace JSC::DFG

namespace WTF {

void printInternal(PrintStream&, JSC::DFG::UseKind);

} // namespace WTF

#endif // ENABLE(DFG_JIT)
