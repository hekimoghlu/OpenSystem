/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 13, 2024.
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
#include "BooleanConstructor.h"

#include "BooleanPrototype.h"
#include "JSCInlines.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(BooleanConstructor);

const ClassInfo BooleanConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(BooleanConstructor) };

static JSC_DECLARE_HOST_FUNCTION(callBooleanConstructor);
static JSC_DECLARE_HOST_FUNCTION(constructWithBooleanConstructor);

// ECMA 15.6.1
JSC_DEFINE_HOST_FUNCTION(callBooleanConstructor, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    return JSValue::encode(jsBoolean(callFrame->argument(0).toBoolean(globalObject)));
}

// ECMA 15.6.2
JSC_DEFINE_HOST_FUNCTION(constructWithBooleanConstructor, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSValue boolean = jsBoolean(callFrame->argument(0).toBoolean(globalObject));

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* booleanStructure = JSC_GET_DERIVED_STRUCTURE(vm, booleanObjectStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    BooleanObject* obj = BooleanObject::create(vm, booleanStructure);
    obj->setInternalValue(vm, boolean);
    return JSValue::encode(obj);
}

BooleanConstructor::BooleanConstructor(VM& vm, NativeExecutable* executable, JSGlobalObject* globalObject, Structure* structure)
    : Base(vm, executable, globalObject, structure)
{
}

void BooleanConstructor::finishCreation(VM& vm, BooleanPrototype* booleanPrototype)
{
    Base::finishCreation(vm);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, booleanPrototype, PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum | PropertyAttribute::DontDelete);
    putDirectWithoutTransition(vm, vm.propertyNames->length, jsNumber(1), PropertyAttribute::DontEnum | PropertyAttribute::ReadOnly);
    putDirectWithoutTransition(vm, vm.propertyNames->name, jsString(vm, vm.propertyNames->Boolean.string()), PropertyAttribute::ReadOnly | PropertyAttribute::DontEnum);
}

BooleanConstructor* BooleanConstructor::create(VM& vm, Structure* structure, BooleanPrototype* booleanPrototype)
{
    JSGlobalObject* globalObject = structure->globalObject();
    NativeExecutable* executable = vm.getHostFunction(callBooleanConstructor, ImplementationVisibility::Public, BooleanConstructorIntrinsic, constructWithBooleanConstructor, nullptr, vm.propertyNames->Boolean.string());
    BooleanConstructor* constructor = new (NotNull, allocateCell<BooleanConstructor>(vm)) BooleanConstructor(vm, executable, globalObject, structure);
    constructor->finishCreation(vm, booleanPrototype);
    return constructor;
}

JSObject* constructBooleanFromImmediateBoolean(JSGlobalObject* globalObject, JSValue immediateBooleanValue)
{
    VM& vm = globalObject->vm();
    BooleanObject* obj = BooleanObject::create(vm, globalObject->booleanObjectStructure());
    obj->setInternalValue(vm, immediateBooleanValue);
    return obj;
}

} // namespace JSC
