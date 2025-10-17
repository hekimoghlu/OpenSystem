/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 7, 2025.
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
#include "ShadowRealmConstructor.h"

#include "JSCInlines.h"
#include "ShadowRealmObjectInlines.h"

namespace JSC {

const ClassInfo ShadowRealmConstructor::s_info = { "Function"_s, &InternalFunction::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(ShadowRealmConstructor) };

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(ShadowRealmConstructor);

static JSC_DECLARE_HOST_FUNCTION(callShadowRealm);
static JSC_DECLARE_HOST_FUNCTION(constructWithShadowRealmConstructor);

ShadowRealmConstructor::ShadowRealmConstructor(VM& vm, Structure* structure)
    : InternalFunction(vm, structure, callShadowRealm, constructWithShadowRealmConstructor)
{
}

void ShadowRealmConstructor::finishCreation(VM& vm, ShadowRealmPrototype* shadowRealmPrototype)
{
    Base::finishCreation(vm, 0, vm.propertyNames->ShadowRealm.string(), PropertyAdditionMode::WithoutStructureTransition);
    putDirectWithoutTransition(vm, vm.propertyNames->prototype, shadowRealmPrototype, PropertyAttribute::DontEnum | PropertyAttribute::DontDelete | PropertyAttribute::ReadOnly);
}

JSC_DEFINE_HOST_FUNCTION(constructWithShadowRealmConstructor, (JSGlobalObject* globalObject, CallFrame*))
{
    VM& vm = globalObject->vm();
    Structure* shadowRealmStructure = ShadowRealmObject::createStructure(vm, globalObject, globalObject->shadowRealmPrototype());
    JSObject* shadowRealmObject = ShadowRealmObject::create(vm, shadowRealmStructure, globalObject);
    return JSValue::encode(shadowRealmObject);
}

JSC_DEFINE_HOST_FUNCTION(callShadowRealm, (JSGlobalObject* globalObject, CallFrame*))
{
    auto scope = DECLARE_THROW_SCOPE(globalObject->vm());
    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "ShadowRealm"_s));
}

} // namespace JSC
