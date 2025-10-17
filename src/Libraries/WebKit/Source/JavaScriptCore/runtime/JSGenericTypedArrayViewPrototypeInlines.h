/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 6, 2022.
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

#include "JSGenericTypedArrayViewPrototype.h"

#include "JSTypedArrays.h"

namespace JSC {
    
template<typename ViewClass>
JSGenericTypedArrayViewPrototype<ViewClass>::JSGenericTypedArrayViewPrototype(VM& vm, Structure* structure)
    : Base(vm, structure)
{
}

template<typename ViewClass>
void JSGenericTypedArrayViewPrototype<ViewClass>::finishCreation(
    VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    
    ASSERT(inherits(info()));

    putDirectWithoutTransition(vm, vm.propertyNames->BYTES_PER_ELEMENT, jsNumber(ViewClass::elementSize), PropertyAttribute::DontEnum | PropertyAttribute::ReadOnly | PropertyAttribute::DontDelete);

    if constexpr (std::is_same_v<ViewClass, JSUint8Array>) {
        if (Options::useUint8ArrayBase64Methods()) {
            JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("setFromBase64"_s, uint8ArrayPrototypeSetFromBase64, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Public);
            JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("setFromHex"_s, uint8ArrayPrototypeSetFromHex, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Public);
            JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("toBase64"_s, uint8ArrayPrototypeToBase64, static_cast<unsigned>(PropertyAttribute::DontEnum), 0, ImplementationVisibility::Public);
            JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("toHex"_s, uint8ArrayPrototypeToHex, static_cast<unsigned>(PropertyAttribute::DontEnum), 0, ImplementationVisibility::Public);
        }
    }

    globalObject->installTypedArrayIteratorProtocolWatchpoint(this, ViewClass::TypedArrayStorageType);
}

template<typename ViewClass>
JSGenericTypedArrayViewPrototype<ViewClass>*
JSGenericTypedArrayViewPrototype<ViewClass>::create(
    VM& vm, JSGlobalObject* globalObject, Structure* structure)
{
    JSGenericTypedArrayViewPrototype* prototype =
        new (NotNull, allocateCell<JSGenericTypedArrayViewPrototype>(vm))
        JSGenericTypedArrayViewPrototype(vm, structure);
    prototype->finishCreation(vm, globalObject);
    return prototype;
}

template<typename ViewClass>
Structure* JSGenericTypedArrayViewPrototype<ViewClass>::createStructure(
    VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(
        vm, globalObject, prototype, TypeInfo(ObjectType, StructureFlags), info());
}

} // namespace JSC
