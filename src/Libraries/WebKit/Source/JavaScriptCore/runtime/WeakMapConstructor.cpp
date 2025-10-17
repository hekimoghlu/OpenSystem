/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 17, 2024.
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
#include "WeakMapConstructor.h"

#include "IteratorOperations.h"
#include "JSCInlines.h"
#include "JSWeakMapInlines.h"
#include "WeakMapPrototype.h"

namespace JSC {

const ClassInfo WeakMapConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WeakMapConstructor) };

void WeakMapConstructor::finishCreation(VM& vm, WeakMapPrototype* prototype)
{
    Base::finishCreation(vm, 0, "WeakMap"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
}

static JSC_DECLARE_HOST_FUNCTION(callWeakMap);
static JSC_DECLARE_HOST_FUNCTION(constructWeakMap);

WeakMapConstructor::WeakMapConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callWeakMap, constructWeakMap)
{
}

JSC_DEFINE_HOST_FUNCTION(callWeakMap, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "WeakMap"_s));
}

JSC_DEFINE_HOST_FUNCTION(constructWeakMap, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* weakMapStructure = JSC_GET_DERIVED_STRUCTURE(vm, weakMapStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    JSWeakMap* weakMap = JSWeakMap::create(vm, weakMapStructure);
    JSValue iterable = callFrame->argument(0);
    if (iterable.isUndefinedOrNull())
        return JSValue::encode(weakMap);

    JSValue adderFunction = weakMap->JSObject::get(globalObject, vm.propertyNames->set);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());

    auto adderFunctionCallData = JSC::getCallData(adderFunction);
    if (adderFunctionCallData.type == CallData::Type::None)
        return throwVMTypeError(globalObject, scope, "'set' property of a WeakMap should be callable."_s);

    bool canPerformFastSet = adderFunctionCallData.type == CallData::Type::Native && adderFunctionCallData.native.function == protoFuncWeakMapSet;

    scope.release();
    forEachInIterable(globalObject, iterable, [&](VM& vm, JSGlobalObject* globalObject, JSValue nextItem) {
        auto scope = DECLARE_THROW_SCOPE(vm);
        if (!nextItem.isObject()) {
            throwTypeError(globalObject, scope, "WeakMap requires that an entry be an Object."_s);
            return;
        }

        JSObject* nextObject = asObject(nextItem);

        JSValue key = nextObject->getIndex(globalObject, static_cast<unsigned>(0));
        RETURN_IF_EXCEPTION(scope, void());

        JSValue value = nextObject->getIndex(globalObject, static_cast<unsigned>(1));
        RETURN_IF_EXCEPTION(scope, void());

        if (canPerformFastSet) {
            if (UNLIKELY(!canBeHeldWeakly(key))) {
                throwTypeError(asObject(adderFunction)->globalObject(), scope, WeakMapInvalidKeyError);
                return;
            }
            weakMap->set(vm, key.asCell(), value);
            return;
        }

        MarkedArgumentBuffer arguments;
        arguments.append(key);
        arguments.append(value);
        ASSERT(!arguments.hasOverflowed());
        scope.release();
        call(globalObject, adderFunction, adderFunctionCallData, weakMap, arguments);
    });

    return JSValue::encode(weakMap);
}

}
