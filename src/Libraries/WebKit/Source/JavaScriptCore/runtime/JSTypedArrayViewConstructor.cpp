/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 18, 2022.
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
#include "JSTypedArrayViewConstructor.h"

#include "JSCBuiltins.h"
#include "JSCInlines.h"
#include "JSTypedArrayViewPrototype.h"

namespace JSC {

static JSC_DECLARE_HOST_FUNCTION(constructTypedArrayView);

JSTypedArrayViewConstructor::JSTypedArrayViewConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, constructTypedArrayView, constructTypedArrayView)
{
}

const ClassInfo JSTypedArrayViewConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(JSTypedArrayViewConstructor) };

void JSTypedArrayViewConstructor::finishCreation(VM& vm, JSGlobalObject* globalObject, JSTypedArrayViewPrototype* prototype)
{
    Base::finishCreation(vm, 0, "TypedArray"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    putDirectNonIndexAccessorWithoutTransition(vm, vm.propertyNames->speciesSymbol, globalObject->typedArraySpeciesGetterSetter(), PropertyAttribute::Accessor | PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum);

    JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->of, typedArrayConstructorOfCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
    JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->from, typedArrayConstructorFromCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
    globalObject->installTypedArrayConstructorSpeciesWatchpoint(this);
}

Structure* JSTypedArrayViewConstructor::createStructure(
    VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(InternalFunctionType, StructureFlags), info());
}

JSC_DEFINE_HOST_FUNCTION(constructTypedArrayView, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    return throwVMTypeError(globalObject, scope, "%TypedArray% should not be called directly"_s);
}

} // namespace JSC
