/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 17, 2022.
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
#include "DOMWrapperWorld.h"

#include "CommonVM.h"
#include "WebCoreJSClientData.h"
#include "WindowProxy.h"
#include <JavaScriptCore/HeapCellInlines.h>
#include <JavaScriptCore/SlotVisitorInlines.h>
#include <JavaScriptCore/WeakInlines.h>
#include <wtf/MainThread.h>

namespace WebCore {
using namespace JSC;

DOMWrapperWorld::DOMWrapperWorld(JSC::VM& vm, Type type, const String& name)
    : m_vm(vm)
    , m_name(name)
    , m_type(type)
{
    VM::ClientData* clientData = m_vm.clientData;
    ASSERT(clientData);
    static_cast<JSVMClientData*>(clientData)->rememberWorld(*this);
}

DOMWrapperWorld::~DOMWrapperWorld()
{
    VM::ClientData* clientData = m_vm.clientData;
    ASSERT(clientData);
    static_cast<JSVMClientData*>(clientData)->forgetWorld(*this);

    // These items are created lazily.
    while (!m_jsWindowProxies.isEmpty())
        (*m_jsWindowProxies.begin())->destroyJSWindowProxy(*this);
}

void DOMWrapperWorld::clearWrappers()
{
    m_wrappers.clear();

    // These items are created lazily.
    while (!m_jsWindowProxies.isEmpty())
        (*m_jsWindowProxies.begin())->destroyJSWindowProxy(*this);
}

DOMWrapperWorld& normalWorld(JSC::VM& vm)
{
    VM::ClientData* clientData = vm.clientData;
    ASSERT(clientData);
    return static_cast<JSVMClientData*>(clientData)->normalWorld();
}

DOMWrapperWorld& mainThreadNormalWorld()
{
    ASSERT(isMainThread());
    static DOMWrapperWorld& cachedNormalWorld = normalWorld(commonVM());
    return cachedNormalWorld;
}

} // namespace WebCore
