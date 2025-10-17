/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 1, 2022.
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
#include "JSPromiseConstructor.h"

#include "BuiltinNames.h"
#include "GetterSetter.h"
#include "JSCBuiltins.h"
#include "JSCInlines.h"
#include "JSPromisePrototype.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(JSPromiseConstructor);

}

#include "JSPromiseConstructor.lut.h"

namespace JSC {

const ClassInfo JSPromiseConstructor::s_info = { "Function"_s, &Base::s_info, &promiseConstructorTable, nullptr, CREATE_METHOD_TABLE(JSPromiseConstructor) };

/* Source for JSPromiseConstructor.lut.h
@begin promiseConstructorTable
  resolve         JSBuiltin             DontEnum|Function 1
  reject          JSBuiltin             DontEnum|Function 1
  race            JSBuiltin             DontEnum|Function 1
  all             JSBuiltin             DontEnum|Function 1
  allSettled      JSBuiltin             DontEnum|Function 1
  any             JSBuiltin             DontEnum|Function 1
@end
*/

JSPromiseConstructor* JSPromiseConstructor::create(VM& vm, Structure* structure, JSPromisePrototype* promisePrototype)
{
    JSGlobalObject* globalObject = structure->globalObject();
    FunctionExecutable* executable = promiseConstructorPromiseConstructorCodeGenerator(vm);
    JSPromiseConstructor* constructor = new (NotNull, allocateCell<JSPromiseConstructor>(vm)) JSPromiseConstructor(vm, executable, globalObject, structure);
    constructor->finishCreation(vm, promisePrototype);
    constructor->addOwnInternalSlots(vm, globalObject);
    return constructor;
}

Structure* JSPromiseConstructor::createStructure(VM& vm, JSGlobalObject* globalObject, JSValue prototype)
{
    return Structure::create(vm, globalObject, prototype, TypeInfo(JSFunctionType, StructureFlags), info());
}

JSPromiseConstructor::JSPromiseConstructor(VM& vm, FunctionExecutable* executable, JSGlobalObject* globalObject, Structure* structure)
    : Base(vm, executable, globalObject, structure)
{
}

void JSPromiseConstructor::finishCreation(VM& vm, JSPromisePrototype* promisePrototype)
{
    Base::finishCreation(vm);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, promisePrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    putDirectWithoutTransition(vm, vm.propertyNames->length, jsNumber(1), PropertyAttribute::DontEnum | PropertyAttribute::ReadOnly);

    JSGlobalObject* globalObject = this->globalObject();

    GetterSetter* speciesGetterSetter = GetterSetter::create(vm, globalObject, JSFunction::create(vm, globalObject, 0, "get [Symbol.species]"_s, globalFuncSpeciesGetter, ImplementationVisibility::Public, SpeciesGetterIntrinsic), nullptr);
    putDirectNonIndexAccessorWithoutTransition(vm, vm.propertyNames->speciesSymbol, speciesGetterSetter, PropertyAttribute::Accessor | PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum);
    JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().withResolversPublicName(), promiseConstructorWithResolversCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
    if (Options::usePromiseTryMethod())
        JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->tryKeyword, promiseConstructorTryCodeGenerator, static_cast<unsigned>(PropertyAttribute::DontEnum));
}

void JSPromiseConstructor::addOwnInternalSlots(VM& vm, JSGlobalObject* globalObject)
{
    JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().resolvePrivateName(), promiseConstructorResolveCodeGenerator, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
    JSC_BUILTIN_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->builtinNames().rejectPrivateName(), promiseConstructorRejectCodeGenerator, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
}

} // namespace JSC
