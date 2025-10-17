/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 9, 2022.
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
#include "ProxyConstructor.h"

#include "JSCInlines.h"
#include "ObjectConstructor.h"
#include "ProxyObject.h"
#include "ProxyRevoke.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(ProxyConstructor);

const ClassInfo ProxyConstructor::s_info = { "Proxy"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(ProxyConstructor) };

ProxyConstructor* ProxyConstructor::create(VM& vm, Structure* structure)
{
    ProxyConstructor* constructor = new (NotNull, allocateCell<ProxyConstructor>(vm)) ProxyConstructor(vm, structure);
    constructor->finishCreation(vm, structure->globalObject());
    return constructor;
}

static JSC_DECLARE_HOST_FUNCTION(callProxy);
static JSC_DECLARE_HOST_FUNCTION(constructProxyObject);
static JSC_DECLARE_HOST_FUNCTION(makeRevocableProxy);

ProxyConstructor::ProxyConstructor(VM& vm, Structure* structure)
    : Base(vm, structure, callProxy, constructProxyObject)
{
}

JSC_DEFINE_HOST_FUNCTION(makeRevocableProxy, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    VM& vm = globalObject->vm();
    auto scope = DECLARE_THROW_SCOPE(vm);
    if (callFrame->argumentCount() < 2)
        return throwVMTypeError(globalObject, scope, "Proxy.revocable needs to be called with two arguments: the target and the handler"_s);

    JSValue target = callFrame->argument(0);
    JSValue handler = callFrame->argument(1);
    ProxyObject* proxy = ProxyObject::create(globalObject, target, handler);
    RETURN_IF_EXCEPTION(scope, encodedJSValue());
    ProxyRevoke* revoke = ProxyRevoke::create(vm, globalObject->proxyRevokeStructure(), proxy);
    scope.assertNoException();

    JSObject* result = constructEmptyObject(globalObject);
    result->putDirect(vm, makeIdentifier(vm, "proxy"_s), proxy, static_cast<unsigned>(PropertyAttribute::None));
    result->putDirect(vm, makeIdentifier(vm, "revoke"_s), revoke, static_cast<unsigned>(PropertyAttribute::None));

    return JSValue::encode(result);
}

void ProxyConstructor::finishCreation(VM& vm, JSGlobalObject* globalObject)
{
    Base::finishCreation(vm, 2, "Proxy"_s, PropertyAdditionMode::WithoutStructureTransition);
    JSC_NATIVE_FUNCTION_WITHOUT_TRANSITION("revocable"_s, makeRevocableProxy, static_cast<unsigned>(PropertyAttribute::DontEnum), 2, ImplementationVisibility::Public);
}

JSC_DEFINE_HOST_FUNCTION(constructProxyObject, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    JSValue target = callFrame->argument(0);
    JSValue handler = callFrame->argument(1);
    return JSValue::encode(ProxyObject::create(globalObject, target, handler));
}

JSC_DEFINE_HOST_FUNCTION(callProxy, (JSGlobalObject* globalObject, CallFrame*))
{
    auto scope = DECLARE_THROW_SCOPE(globalObject->vm());
    return JSValue::encode(throwConstructorCannotBeCalledAsFunctionTypeError(globalObject, scope, "Proxy"_s));
}

} // namespace JSC
