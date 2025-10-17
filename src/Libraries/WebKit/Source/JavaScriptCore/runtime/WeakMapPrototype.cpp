/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 7, 2023.
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
#include "WeakMapPrototype.h"

#include "CachedCall.h"
#include "InterpreterInlines.h"
#include "JSCInlines.h"
#include "JSWeakMapInlines.h"
#include "VMEntryScopeInlines.h"

namespace JSC {

const ASCIILiteral WeakMapInvalidKeyError { "WeakMap keys must be objects or non-registered symbols"_s };

const ClassInfo WeakMapPrototype::s_info = { "WeakMap"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WeakMapPrototype) };

static JSC_DECLARE_HOST_FUNCTION(protoFuncWeakMapDelete);
static JSC_DECLARE_HOST_FUNCTION(protoFuncWeakMapGet);
static JSC_DECLARE_HOST_FUNCTION(protoFuncWeakMapHas);
static JSC_DECLARE_HOST_FUNCTION(protoFuncWeakMapGetOrInsert);
static JSC_DECLARE_HOST_FUNCTION(protoFuncWeakMapGetOrInsertComputed);

void WeakMapPrototype::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));

    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->deleteKeyword, protoFuncWeakMapDelete, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Public);
    JSC_NATIVE_INTRINSIC_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->get, protoFuncWeakMapGet, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Public, JSWeakMapGetIntrinsic);
    JSC_NATIVE_INTRINSIC_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->has, protoFuncWeakMapHas, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Public, JSWeakMapHasIntrinsic);
    JSC_NATIVE_INTRINSIC_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->set, protoFuncWeakMapSet, static_cast<unsigned>(PropertyAttribute::DontEnum), 2, ImplementationVisibility::Public, JSWeakMapSetIntrinsic);

    if (Options::useMapGetOrInsert()) {
        JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("getOrInsert"_s, protoFuncWeakMapGetOrInsert, static_cast<unsigned>(PropertyAttribute::DontEnum), 2, ImplementationVisibility::Public);
        JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("getOrInsertComputed"_s, protoFuncWeakMapGetOrInsertComputed, static_cast<unsigned>(PropertyAttribute::DontEnum), 2, ImplementationVisibility::Public);
    }

    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

ALWAYS_INLINE static JSWeakMap* getWeakMap(JSGlobalObject* globalObject, JSValue value)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (UNLIKELY(!value.isObject())) {
        throwTypeError(globalObject, scope, "Called WeakMap function on non-object"_s);
        return nullptr;
    }

    auto* map = jsDynamicCast<JSWeakMap*>(asObject(value));
    if (LIKELY(map))
        return map;

    throwTypeError(globalObject, scope, "Called WeakMap function on a non-WeakMap object"_s);
    return nullptr;
}

JSC_DEFINE_HOST_FUNCTION(protoFuncWeakMapDelete, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    auto* map = getWeakMap(globalObject, callFrame->thisValue());
    if (!map)
        return JSValue::encode(jsUndefined());
    JSValue key = callFrame->argument(0);
    if (UNLIKELY(!key.isCell()))
        return JSValue::encode(jsBoolean(false));
    return JSValue::encode(jsBoolean(map->remove(key.asCell())));
}

JSC_DEFINE_HOST_FUNCTION(protoFuncWeakMapGet, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    auto* map = getWeakMap(globalObject, callFrame->thisValue());
    if (!map)
        return JSValue::encode(jsUndefined());
    JSValue key = callFrame->argument(0);
    if (UNLIKELY(!key.isCell()))
        return JSValue::encode(jsUndefined());
    return JSValue::encode(map->get(key.asCell()));
}

JSC_DEFINE_HOST_FUNCTION(protoFuncWeakMapHas, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    auto* map = getWeakMap(globalObject, callFrame->thisValue());
    if (!map)
        return JSValue::encode(jsUndefined());
    JSValue key = callFrame->argument(0);
    if (UNLIKELY(!key.isCell()))
        return JSValue::encode(jsBoolean(false));
    return JSValue::encode(jsBoolean(map->has(key.asCell())));
}

JSC_DEFINE_HOST_FUNCTION(protoFuncWeakMapSet, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* map = getWeakMap(globalObject, callFrame->thisValue());
    EXCEPTION_ASSERT(!!scope.exception() == !map);
    if (!map)
        return JSValue::encode(jsUndefined());
    JSValue key = callFrame->argument(0);
    if (UNLIKELY(!canBeHeldWeakly(key)))
        return throwVMTypeError(globalObject, scope, WeakMapInvalidKeyError);
    map->set(vm, key.asCell(), callFrame->argument(1));
    return JSValue::encode(callFrame->thisValue());
}

JSC_DEFINE_HOST_FUNCTION(protoFuncWeakMapGetOrInsert, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* map = getWeakMap(globalObject, callFrame->thisValue());
    EXCEPTION_ASSERT(!!scope.exception() == !map);
    if (!map)
        return JSValue::encode(jsUndefined());

    JSValue key = callFrame->argument(0);
    if (UNLIKELY(!canBeHeldWeakly(key)))
        return throwVMTypeError(globalObject, scope, WeakMapInvalidKeyError);

    JSCell* keyCell = key.asCell();

    auto hash = jsWeakMapHash(keyCell);

    JSValue value;

    {
        DisallowGC disallowGC;

        auto [index, exists] = map->findBucketIndex(keyCell, hash);
        if (exists)
            value = map->getBucket(keyCell, hash, index);
        else {
            value = callFrame->argument(1);
            map->addBucket(vm, keyCell, value, hash, index);
        }
    }

    return JSValue::encode(value);
}

JSC_DEFINE_HOST_FUNCTION(protoFuncWeakMapGetOrInsertComputed, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* map = getWeakMap(globalObject, callFrame->thisValue());
    EXCEPTION_ASSERT(!!scope.exception() == !map);
    if (!map)
        return JSValue::encode(jsUndefined());

    JSValue key = callFrame->argument(0);
    if (UNLIKELY(!canBeHeldWeakly(key)))
        return throwVMTypeError(globalObject, scope, WeakMapInvalidKeyError);

    JSValue valueCallback = callFrame->argument(1);
    if (!valueCallback.isCallable())
        return throwVMTypeError(globalObject, scope, "WeakMap.prototype.getOrInsertComputed requires the callback argument to be callable."_s);

    JSCell* keyCell = key.asCell();

    auto hash = jsWeakMapHash(keyCell);

    JSValue value;

    {
        DisallowGC disallowGC;

        auto [index, exists] = map->findBucketIndex(keyCell, hash);
        if (exists)
            value = map->getBucket(keyCell, hash, index);
        else {
            auto callData = JSC::getCallData(valueCallback);
            ASSERT(callData.type != CallData::Type::None);

            if (LIKELY(callData.type == CallData::Type::JS)) {
                CachedCall cachedCall(globalObject, jsCast<JSFunction*>(valueCallback), 2);
                RETURN_IF_EXCEPTION(scope, { });

                value = cachedCall.callWithArguments(globalObject, jsUndefined(), key);
                RETURN_IF_EXCEPTION(scope, { });
            } else {
                MarkedArgumentBuffer args;
                args.append(key);
                ASSERT(!args.hasOverflowed());

                value = call(globalObject, valueCallback, callData, jsUndefined(), args);
                RETURN_IF_EXCEPTION(scope, { });
            }

            map->addBucket(vm, keyCell, value, hash, index);
        }
    }

    return JSValue::encode(value);
}

}
