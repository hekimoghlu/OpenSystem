/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 15, 2024.
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
#include "WeakSetPrototype.h"

#include "HashMapHelper.h"
#include "JSCInlines.h"
#include "JSWeakSet.h"
#include "WeakMapImplInlines.h"

namespace JSC {

const ASCIILiteral WeakSetInvalidValueError { "WeakSet values must be objects or non-registered symbols"_s };

const ClassInfo WeakSetPrototype::s_info = { "WeakSet"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WeakSetPrototype) };

static JSC_DECLARE_HOST_FUNCTION(protoFuncWeakSetDelete);
static JSC_DECLARE_HOST_FUNCTION(protoFuncWeakSetHas);

void WeakSetPrototype::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));

    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->deleteKeyword, protoFuncWeakSetDelete, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Public);
    JSC_NATIVE_INTRINSIC_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->has, protoFuncWeakSetHas, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Public, JSWeakSetHasIntrinsic);
    JSC_NATIVE_INTRINSIC_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->add, protoFuncWeakSetAdd, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Public, JSWeakSetAddIntrinsic);

    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

ALWAYS_INLINE static JSWeakSet* getWeakSet(JSGlobalObject* globalObject, JSValue value)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (UNLIKELY(!value.isObject())) {
        throwTypeError(globalObject, scope, "Called WeakSet function on non-object"_s);
        return nullptr;
    }

    auto* set = jsDynamicCast<JSWeakSet*>(asObject(value));
    if (LIKELY(set))
        return set;

    throwTypeError(globalObject, scope, "Called WeakSet function on a non-WeakSet object"_s);
    return nullptr;
}

JSC_DEFINE_HOST_FUNCTION(protoFuncWeakSetDelete, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    auto* set = getWeakSet(globalObject, callFrame->thisValue());
    if (!set)
        return JSValue::encode(jsUndefined());
    JSValue key = callFrame->argument(0);
    if (UNLIKELY(!key.isCell()))
        return JSValue::encode(jsBoolean(false));
    return JSValue::encode(jsBoolean(set->remove(key.asCell())));
}

JSC_DEFINE_HOST_FUNCTION(protoFuncWeakSetHas, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    auto* set = getWeakSet(globalObject, callFrame->thisValue());
    if (!set)
        return JSValue::encode(jsUndefined());
    JSValue key = callFrame->argument(0);
    if (UNLIKELY(!key.isCell()))
        return JSValue::encode(jsBoolean(false));
    return JSValue::encode(jsBoolean(set->has(key.asCell())));
}

JSC_DEFINE_HOST_FUNCTION(protoFuncWeakSetAdd, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* set = getWeakSet(globalObject, callFrame->thisValue());
    EXCEPTION_ASSERT(!!scope.exception() == !set);
    if (!set)
        return JSValue::encode(jsUndefined());
    JSValue key = callFrame->argument(0);
    if (UNLIKELY(!canBeHeldWeakly(key)))
        return throwVMTypeError(globalObject, scope, WeakSetInvalidValueError);
    set->add(vm, key.asCell());
    return JSValue::encode(callFrame->thisValue());
}

}
