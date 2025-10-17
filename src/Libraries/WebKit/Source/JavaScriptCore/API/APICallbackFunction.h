/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 12, 2023.
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
#ifndef APICallbackFunction_h
#define APICallbackFunction_h

#include "APICast.h"
#include "Error.h"
#include "JSCallbackConstructor.h"
#include "JSLock.h"
#include <wtf/Vector.h>

namespace JSC {

struct APICallbackFunction {
    template <typename T> static EncodedJSValue callImpl(JSGlobalObject*, CallFrame*);
    template <typename T> static EncodedJSValue constructImpl(JSGlobalObject*, CallFrame*);
};

template <typename T>
EncodedJSValue APICallbackFunction::callImpl(JSGlobalObject* globalObject, CallFrame* callFrame)
{
    VM& vm = getVM(globalObject);
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSContextRef execRef = toRef(globalObject);
    JSObjectRef functionRef = toRef(callFrame->jsCallee());
    JSObjectRef thisObjRef = toRef(jsCast<JSObject*>(callFrame->thisValue().toThis(globalObject, ECMAMode::sloppy())));

    int argumentCount = static_cast<int>(callFrame->argumentCount());
    Vector<JSValueRef, 16> arguments(argumentCount, [&](size_t i) {
        return toRef(globalObject, callFrame->uncheckedArgument(i));
    });

    JSValueRef exception = nullptr;
    JSValueRef result;
    {
        JSLock::DropAllLocks dropAllLocks(globalObject);
        result = jsCast<T*>(toJS(functionRef))->functionCallback()(execRef, functionRef, thisObjRef, argumentCount, arguments.data(), &exception);
    }
    if (exception) {
        throwException(globalObject, scope, toJS(globalObject, exception));
        return JSValue::encode(jsUndefined());
    }

    // result must be a valid JSValue.
    if (!result)
        return JSValue::encode(jsUndefined());

    return JSValue::encode(toJS(globalObject, result));
}

template <typename T>
EncodedJSValue APICallbackFunction::constructImpl(JSGlobalObject* globalObject, CallFrame* callFrame)
{
    VM& vm = getVM(globalObject);
    auto scope = DECLARE_THROW_SCOPE(vm);
    JSValue callee = callFrame->jsCallee();
    T* constructor = jsCast<T*>(callFrame->jsCallee());
    JSContextRef ctx = toRef(globalObject);
    JSObjectRef constructorRef = toRef(constructor);

    JSObjectCallAsConstructorCallback callback = constructor->constructCallback();
    if (callback) {
        JSValue prototype;
        JSValue newTarget = callFrame->newTarget();
        // If we are doing a derived class construction get the .prototype property off the new target first so we behave closer to normal JS.
        if (newTarget != constructor) {
            prototype = newTarget.get(globalObject, vm.propertyNames->prototype);
            RETURN_IF_EXCEPTION(scope, { });
        }

        size_t argumentCount = callFrame->argumentCount();
        Vector<JSValueRef, 16> arguments(argumentCount, [&](size_t i) {
            return toRef(globalObject, callFrame->uncheckedArgument(i));
        });

        JSValueRef exception = nullptr;
        JSObjectRef result;
        {
            JSLock::DropAllLocks dropAllLocks(globalObject);
            result = callback(ctx, constructorRef, argumentCount, arguments.data(), &exception);
        }

        if (exception) {
            throwException(globalObject, scope, toJS(globalObject, exception));
            return JSValue::encode(jsUndefined());
        }
        // result must be a valid JSValue.
        if (!result)
            return throwVMTypeError(globalObject, scope);

        JSObject* newObject = toJS(result);
        // This won't trigger proxy traps on newObject's prototype handler but that's probably desirable here anyway.
        if (newTarget != constructor && newObject->getPrototypeDirect() == constructor->get(globalObject, vm.propertyNames->prototype)) {
            RETURN_IF_EXCEPTION(scope, { });
            newObject->setPrototype(vm, globalObject, prototype);
            RETURN_IF_EXCEPTION(scope, { });
        }

        return JSValue::encode(newObject);
    }
    
    return JSValue::encode(toJS(JSObjectMake(ctx, jsCast<JSCallbackConstructor*>(callee)->classRef(), nullptr)));
}

} // namespace JSC

#endif // APICallbackFunction_h
