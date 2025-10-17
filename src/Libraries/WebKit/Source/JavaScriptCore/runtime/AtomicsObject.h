/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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

#include "JSObject.h"

namespace JSC {

class AtomicsObject final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    template<typename CellType, SubspaceAccess>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(AtomicsObject, Base);
        return &vm.plainObjectSpace();
    }
    
    static AtomicsObject* create(VM&, JSGlobalObject*, Structure*);
    
    DECLARE_INFO;
    
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    AtomicsObject(VM&, Structure*);
    void finishCreation(VM&, JSGlobalObject*);
};

JSC_DECLARE_JIT_OPERATION(operationAtomicsAdd, EncodedJSValue, (JSGlobalObject*, EncodedJSValue base, EncodedJSValue index, EncodedJSValue operand));
JSC_DECLARE_JIT_OPERATION(operationAtomicsAnd, EncodedJSValue, (JSGlobalObject*, EncodedJSValue base, EncodedJSValue index, EncodedJSValue operand));
JSC_DECLARE_JIT_OPERATION(operationAtomicsCompareExchange, EncodedJSValue, (JSGlobalObject*, EncodedJSValue base, EncodedJSValue index, EncodedJSValue expected, EncodedJSValue newValue));
JSC_DECLARE_JIT_OPERATION(operationAtomicsExchange, EncodedJSValue, (JSGlobalObject*, EncodedJSValue base, EncodedJSValue index, EncodedJSValue operand));
JSC_DECLARE_JIT_OPERATION(operationAtomicsIsLockFree, EncodedJSValue, (JSGlobalObject*, EncodedJSValue size));
JSC_DECLARE_JIT_OPERATION(operationAtomicsLoad, EncodedJSValue, (JSGlobalObject*, EncodedJSValue base, EncodedJSValue index));
JSC_DECLARE_JIT_OPERATION(operationAtomicsOr, EncodedJSValue, (JSGlobalObject*, EncodedJSValue base, EncodedJSValue index, EncodedJSValue operand));
JSC_DECLARE_JIT_OPERATION(operationAtomicsStore, EncodedJSValue, (JSGlobalObject*, EncodedJSValue base, EncodedJSValue index, EncodedJSValue operand));
JSC_DECLARE_JIT_OPERATION(operationAtomicsSub, EncodedJSValue, (JSGlobalObject*, EncodedJSValue base, EncodedJSValue index, EncodedJSValue operand));
JSC_DECLARE_JIT_OPERATION(operationAtomicsXor, EncodedJSValue, (JSGlobalObject*, EncodedJSValue base, EncodedJSValue index, EncodedJSValue operand));

JS_EXPORT_PRIVATE EncodedJSValue getWaiterListSize(JSGlobalObject*, CallFrame*);

} // namespace JSC

