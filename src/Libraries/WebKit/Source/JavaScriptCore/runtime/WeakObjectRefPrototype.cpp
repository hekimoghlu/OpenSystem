/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 25, 2025.
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
#include "WeakObjectRefPrototype.h"

#include "JSCInlines.h"
#include "JSWeakObjectRef.h"

namespace JSC {

const ClassInfo WeakObjectRefPrototype::s_info = { "WeakRef"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WeakObjectRefPrototype) };

static JSC_DECLARE_HOST_FUNCTION(protoFuncWeakRefDeref);

void WeakObjectRefPrototype::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm);
    ASSERT(inherits(info()));

    // FIXME: It wouldn't be hard to make this an intrinsic.
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION(vm.propertyNames->deref, protoFuncWeakRefDeref, static_cast<unsigned>(PropertyAttribute::DontEnum), 0, ImplementationVisibility::Public);

    JSC_TO_STRING_TAG_WITHOUT_TRANSITION();
}

ALWAYS_INLINE static JSWeakObjectRef* getWeakRef(JSGlobalObject* globalObject, JSValue value)
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    if (UNLIKELY(!value.isObject())) {
        throwTypeError(globalObject, scope, "Called WeakRef function on non-object"_s);
        return nullptr;
    }

    auto* ref = jsDynamicCast<JSWeakObjectRef*>(asObject(value));
    if (LIKELY(ref))
        return ref;

    throwTypeError(globalObject, scope, "Called WeakRef function on a non-WeakRef object"_s);
    return nullptr;
}

JSC_DEFINE_HOST_FUNCTION(protoFuncWeakRefDeref, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto* ref = getWeakRef(globalObject, callFrame->thisValue());
    if (!ref)
        return JSValue::encode(jsUndefined());

    auto* value = ref->deref(vm);
    return value ? JSValue::encode(value) : JSValue::encode(jsUndefined());
}

}
