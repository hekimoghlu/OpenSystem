/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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
#include "WeakObjectRefConstructor.h"

#include "JSCInlines.h"
#include "JSWeakObjectRef.h"
#include "WeakMapImplInlines.h"
#include "WeakObjectRefPrototype.h"

namespace JSC {

const ClassInfo WeakObjectRefConstructor::s_info = { "Function"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(WeakObjectRefConstructor) };

void WeakObjectRefConstructor::finishCreation(VM& vm, WeakObjectRefPrototype* prototype)
{
    Base::finishCreation(vm, 1, "WeakRef"_s, PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, prototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
}

static JSC_DECLARE_HOST_FUNCTION(callWeakRef);
static JSC_DECLARE_HOST_FUNCTION(constructWeakRef);

WeakObjectRefConstructor::WeakObjectRefConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callWeakRef, constructWeakRef)
{
}

JSC_DEFINE_HOST_FUNCTION(callWeakRef, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "WeakRef"_s));
}

JSC_DEFINE_HOST_FUNCTION(constructWeakRef, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);

    JSValue target = callFrame->argument(0);
    if (UNLIKELY(!canBeHeldWeakly(target)))
        return throwVMTypeError(globalObject, scope, "First argument to WeakRef should be an object or a non-registered symbol"_s);

    JSObject* newTarget = asObject(callFrame->newTarget());
    Structure* weakObjectRefStructure = JSC_GET_DERIVED_STRUCTURE(vm, weakObjectRefStructure, newTarget, callFrame->jsCallee());
    RETURN_IF_EXCEPTION(scope, { });

    RELEASE_AND_RETURN(scope, JSValue::encode(JSWeakObjectRef::create(vm, weakObjectRefStructure, target.asCell())));
}

}

