/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 2, 2022.
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
#include "WeakSetConstructor.h"

#include "IteratorOperations.h"
#include "JSCInlines.h"
#include "JSWeakSet.h"
#include "WeakMapImplInlines.h"
#include "WeakSetPrototype.h"

namespace JSC {

const ClassInfo WeakSetConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WeakSetConstructor) };

void WeakSetConstructor::finishCreation(VM& vm, WeakSetPrototype* prototype)
{
    Base::finishCreation(vm, 0, "WeakSet"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
}

static JSC_DECLARE_HOST_FUNCTION(callWeakSet);
static JSC_DECLARE_HOST_FUNCTION(constructWeakSet);

WeakSetConstructor::WeakSetConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callWeakSet, constructWeakSet)
{
}

JSC_DEFINE_HOST_FUNCTION(callWeakSet, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "WeakSet"_s));
}

JSC_DEFINE_HOST_FUNCTION(constructWeakSet, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* weakSetStructure = JSC_GET_DERIVED_STRUCTURE(vm, weakSetStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    JSWeakSet* weakSet = JSWeakSet::create(vm, weakSetStructure);
    JSValue iterable = callFrame->argument(0);
    if (iterable.isUndefinedOrNull())
        return JSValue::encode(weakSet);

    JSValue adderFunction = weakSet->JSObject::get(globalObject, vm.propertyNames->add);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    auto adderFunctionCallData = JSC::getCallData(adderFunction);
    if (adderFunctionCallData.type == CallData::Type::None)
        return throwVMTypeError(globalObject, scope, "'add' property of a WeakSet should be callable."_s);

    bool canPerformFastAdd = adderFunctionCallData.type == CallData::Type::Native && adderFunctionCallData.native.function == protoFuncWeakSetAdd;

    scope.release();
    forEachInIterable(globalObject, iterable, [&](VM&, JSGlobalObject* globalObject, JSValue nextValue) {
        if (canPerformFastAdd) {
            if (UNLIKELY(!canBeHeldWeakly(nextValue))) {
                throwTypeError(asObject(adderFunction)->globalObject(), scope, WeakSetInvalidValueError);
                return;
            }
            weakSet->add(vm, nextValue.asCell());
            return;
        }

        MarkedArgumentBuffer arguments;
        arguments.append(nextValue);
        ASSERT(!arguments.hasOverflowed());
        call(globalObject, adderFunction, adderFunctionCallData, weakSet, arguments);
    });

    return JSValue::encode(weakSet);
}

}
