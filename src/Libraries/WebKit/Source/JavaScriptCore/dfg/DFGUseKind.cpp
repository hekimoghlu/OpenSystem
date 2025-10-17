/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 16, 2025.
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
#include "DFGUseKind.h"

#if ENABLE(DFG_JIT)

namespace WTF {

using namespace JSC::DFG;

void printInternal(PrintStream& out, UseKind useKind)
{
    switch (useKind) {
    case UntypedUse:
        out.print("Untyped");
        return;
    case Int32Use:
        out.print("Int32");
        return;
    case KnownInt32Use:
        out.print("KnownInt32");
        return;
    case Int52RepUse:
        out.print("Int52Rep");
        return;
    case AnyIntUse:
        out.print("AnyInt");
        return;
    case NumberUse:
        out.print("Number");
        return;
    case RealNumberUse:
        out.print("RealNumber");
        return;
    case DoubleRepUse:
        out.print("DoubleRep");
        return;
    case DoubleRepRealUse:
        out.print("DoubleRepReal");
        return;
    case DoubleRepAnyIntUse:
        out.print("DoubleRepAnyInt");
        return;
    case BooleanUse:
        out.print("Boolean");
        return;
    case KnownBooleanUse:
        out.print("KnownBoolean");
        return;
    case CellUse:
        out.print("Cell");
        return;
    case KnownCellUse:
        out.print("KnownCell");
        return;
    case CellOrOtherUse:
        out.print("CellOrOther");
        return;
    case ObjectUse:
        out.print("Object");
        return;
    case ArrayUse:
        out.print("Array");
        return;
    case FunctionUse:
        out.print("Function");
        return;
    case FinalObjectUse:
        out.print("FinalObject");
        return;
    case RegExpObjectUse:
        out.print("RegExpObject");
        return;
    case PromiseObjectUse:
        out.print("PromiseObject");
        return;
    case ProxyObjectUse:
        out.print("ProxyObject");
        return;
    case GlobalProxyUse:
        out.print("GlobalProxy");
        return;
    case DerivedArrayUse:
        out.print("DerivedArray");
        return;
    case DateObjectUse:
        out.print("DateObject");
        return;
    case MapObjectUse:
        out.print("MapObject");
        return;
    case SetObjectUse:
        out.print("SetObject");
        return;
    case MapIteratorObjectUse:
        out.print("MapIteratorObject");
        return;
    case SetIteratorObjectUse:
        out.print("SetIteratorObject");
        return;
    case WeakMapObjectUse:
        out.print("WeakMapObject");
        return;
    case WeakSetObjectUse:
        out.print("WeakSetObject");
        return;
    case DataViewObjectUse:
        out.print("DataViewObject");
        return;
    case ObjectOrOtherUse:
        out.print("ObjectOrOther");
        return;
    case StringIdentUse:
        out.print("StringIdent");
        return;
    case StringUse:
        out.print("String");
        return;
    case StringOrOtherUse:
        out.print("StringOrOther");
        return;
    case KnownStringUse:
        out.print("KnownString");
        return;
    case KnownPrimitiveUse:
        out.print("KnownPrimitive");
        return;
    case SymbolUse:
        out.print("Symbol");
        return;
    case AnyBigIntUse:
        out.print("AnyBigInt");
        return;
    case HeapBigIntUse:
        out.print("HeapBigInt");
        return;
    case BigInt32Use:
        out.print("BigInt32");
        return;
    case StringObjectUse:
        out.print("StringObject");
        return;
    case StringOrStringObjectUse:
        out.print("StringOrStringObject");
        return;
    case NotStringVarUse:
        out.print("NotStringVar");
        return;
    case NotSymbolUse:
        out.print("NotSymbol");
        return;
    case NotCellUse:
        out.print("NotCell");
        return;
    case NotCellNorBigIntUse:
        out.print("NotCellNorBigInt");
        return;
    case NotDoubleUse:
        out.print("NotDouble");
        return;
    case NeitherDoubleNorHeapBigIntUse:
        out.print("NeitherDoubleNorHeapBigInt");
        return;
    case NeitherDoubleNorHeapBigIntNorStringUse:
        out.print("NeitherDoubleNorHeapBigIntNorString");
        return;
    case KnownOtherUse:
        out.print("KnownOther");
        return;
    case OtherUse:
        out.print("Other");
        return;
    case MiscUse:
        out.print("Misc");
        return;
    case LastUseKind:
        RELEASE_ASSERT_NOT_REACHED();
        return;
    }
    RELEASE_ASSERT_NOT_REACHED();
}

} // namespace WTF

#endif // ENABLE(DFG_JIT)

