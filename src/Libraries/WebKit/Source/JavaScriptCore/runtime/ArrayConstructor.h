/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 21, 2025.
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

#include "InternalFunction.h"
#include "ProxyObject.h"

namespace JSC {

class ArrayAllocationProfile;
class ArrayPrototype;
class JSArray;
class GetterSetter;

extern const ASCIILiteral ArrayInvalidLengthError;

class ArrayConstructor final : public InternalFunction {
public:
    typedef InternalFunction Base;
    static constexpr unsigned StructureFlags = Base::StructureFlags | HasStaticPropertyTable;

    static ArrayConstructor* create(VM& vm, JSGlobalObject* globalObject, Structure* structure, ArrayPrototype* arrayPrototype)
    {
        ArrayConstructor* constructor = new (NotNull, allocateCell<ArrayConstructor>(vm)) ArrayConstructor(vm, structure);
        constructor->finishCreation(vm, globalObject, arrayPrototype);
        return constructor;
    }

    DECLARE_INFO;

    inline static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

private:
    ArrayConstructor(VM&, Structure*);
    void finishCreation(VM&, JSGlobalObject*, ArrayPrototype*);
};
STATIC_ASSERT_ISO_SUBSPACE_SHARABLE(ArrayConstructor, InternalFunction);

JSArray* constructArrayWithSizeQuirk(JSGlobalObject*, ArrayAllocationProfile*, JSValue length, JSValue prototype = JSValue());

JSC_DECLARE_HOST_FUNCTION(arrayConstructorPrivateFuncIsArraySlow);
bool isArraySlow(JSGlobalObject*, ProxyObject* argument);

// ES6 7.2.2
// https://tc39.github.io/ecma262/#sec-isarray
inline bool isArray(JSGlobalObject* globalObject, JSValue argumentValue)
{
    if (!argumentValue.isObject())
        return false;

    JSObject* argument = jsCast<JSObject*>(argumentValue);
    if (argument->type() == ArrayType || argument->type() == DerivedArrayType)
        return true;

    if (argument->type() != ProxyObjectType)
        return false;
    return isArraySlow(globalObject, jsCast<ProxyObject*>(argument));
}

} // namespace JSC
