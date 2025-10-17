/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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
#include "ProxyRevoke.h"

#include "JSCInlines.h"
#include "ProxyObject.h"

namespace JSC {

STATIC_ASSERT_IS_TRIVIALLY_DESTRUCTIBLE(ProxyRevoke);

const ClassInfo ProxyRevoke::s_info = { "ProxyRevoke"_s, &Base::s_info, nullptr, nullptr, CREATE_METHOD_TABLE(ProxyRevoke) };

ProxyRevoke* ProxyRevoke::create(VM& vm, Structure* structure, ProxyObject* proxy)
{
    ProxyRevoke* revoke = new (NotNull, allocateCell<ProxyRevoke>(vm)) ProxyRevoke(vm, structure, proxy);
    revoke->finishCreation(vm);
    return revoke;
}

static JSC_DECLARE_HOST_FUNCTION(performProxyRevoke);

ProxyRevoke::ProxyRevoke(VM& vm, Structure* structure, ProxyObject* proxy)
    : Base(vm, structure, performProxyRevoke, nullptr)
    , m_proxy(proxy, WriteBarrierEarlyInit)
{
}

void ProxyRevoke::finishCreation(VM& vm)
{
    Base::finishCreation(vm, 0, emptyString());
}

JSC_DEFINE_HOST_FUNCTION(performProxyRevoke, (JSGlobalObject* globalObject, CallFrame* callFrame))
{
    ProxyRevoke* proxyRevoke = jsCast<ProxyRevoke*>(callFrame->jsCallee());
    JSValue proxyValue = proxyRevoke->proxy();
    if (proxyValue.isNull())
        return JSValue::encode(jsUndefined());

    ProxyObject* proxy = jsCast<ProxyObject*>(proxyValue);
    VM& vm = globalObject->vm();
    proxy->revoke(vm);
    proxyRevoke->setProxyToNull(vm);
    return JSValue::encode(jsUndefined());
}

template<typename Visitor>
void ProxyRevoke::visitChildrenImpl(JSCell* cell, Visitor& visitor)
{
    ProxyRevoke* thisObject = jsCast<ProxyRevoke*>(cell);
    ASSERT_GC_OBJECT_INHERITS(thisObject, info());
    Base::visitChildren(thisObject, visitor);

    visitor.append(thisObject->m_proxy);
}

DEFINE_VISIT_CHILDREN(ProxyRevoke);

} // namespace JSC
