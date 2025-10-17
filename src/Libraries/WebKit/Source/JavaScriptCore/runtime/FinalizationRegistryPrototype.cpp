/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 7, 2024.
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
#include "FinalizationRegistryPrototype.h"

#include "Error.h"
#include "JSCInlines.h"
#include "JSFinalizationRegistry.h"
#include "WeakMapImplInlines.h"

namespace JSC {

const ClassInfo FinalizationRegistryPrototype::s_info = { "FinalizationRegistry"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(FinalizationRegistryPrototype) };

static JSC_DECLARE_HOST_FUNCTION(protoFuncFinalizationRegistryRegister);
static JSC_DECLARE_HOST_FUNCTION(protoFuncFinalizationRegistryUnregister);

void FinalizationRegistryPrototype::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));

    // We can't make this a property name because it's a resevered word in C++...
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION(Identifier::fromString(vm, "register"_s), protoFuncFinalizationRegistryRegister, static_cast<unsigned>(PropertyAttribute::DontEnum), 2, ImplementationVisibility::Public);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION(Identifier::fromString(vm, "unregister"_s), protoFuncFinalizationRegistryUnregister, static_cast<unsigned>(PropertyAttribute::DontEnum), 1, ImplementationVisibility::Public);
    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

ALWAYS_INLINE static JSFinalizationRegistry* getFinalizationRegistry(VM& vm, JSGlobalObject* globalObject, JSValue value)
{
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (UNLIKELY(!value.isObject())) {
        throwTypeError(globalObject, scope, "Called FinalizationRegistry function on non-object"_s);
        return nullptr;
    }

    auto* group = jsDynamicCast<JSFinalizationRegistry*>(asObject(value));
    if (LIKELY(group))
        return group;

    throwTypeError(globalObject, scope, "Called FinalizationRegistry function on a non-FinalizationRegistry object"_s);
    return nullptr;
}

JSC_DEFINE_HOST_FUNCTION(protoFuncFinalizationRegistryRegister, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* group = getFinalizationRegistry(vm, globalObject, callFrame->thisValue());
    RETURN_IF_EXCEPTION(scope, { });

    JSValue target = callFrame->argument(0);
    if (UNLIKELY(!canBeHeldWeakly(target)))
        return throwVMTypeError(globalObject, scope, "register requires an object or a non-registered symbol as the target"_s);

    JSValue holdings = callFrame->argument(1);
    if (UNLIKELY(target == holdings))
        return throwVMTypeError(globalObject, scope, "register expects the target object and the holdings parameter are not the same. Otherwise, the target can never be collected"_s);

    JSValue unregisterToken = callFrame->argument(2);
    if (UNLIKELY(!unregisterToken.isUndefined() && !canBeHeldWeakly(unregisterToken)))
        return throwVMTypeError(globalObject, scope, "register requires an object or a non-registered symbol as the unregistration token"_s);

    group->registerTarget(vm, target.asCell(), holdings, unregisterToken);
    return encodedJSUndefined();
}

JSC_DEFINE_HOST_FUNCTION(protoFuncFinalizationRegistryUnregister, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    auto* group = getFinalizationRegistry(vm, globalObject, callFrame->thisValue());
    RETURN_IF_EXCEPTION(scope, { });

    JSValue token = callFrame->argument(0);
    if (UNLIKELY(!canBeHeldWeakly(token)))
        return throwVMTypeError(globalObject, scope, "unregister requires an object or a non-registered symbol as the unregistration token"_s);

    bool result = group->unregister(vm, token.asCell());
    return JSValue::encode(jsBoolean(result));
}

}

