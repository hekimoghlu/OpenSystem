/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 28, 2022.
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
#include "InjectedScriptHost.h"

#include "JSInjectedScriptHost.h"
#include "JSObjectInlines.h"
#include "StructureInlines.h"

namespace Inspector {

using namespace JSC;

InjectedScriptHost::InjectedScriptHost() = default;

InjectedScriptHost::~InjectedScriptHost() = default;

JSValue InjectedScriptHost::wrapper(JSGlobalObject* globalObject)
{
    JSValue value = m_wrappers.getWrapper(globalObject);
    if (value)
        return value;

    VM& vm = globalObject->vm();
    JSObject* prototype = JSInjectedScriptHost::createPrototype(vm, globalObject);
    Structure* structure = JSInjectedScriptHost::createStructure(vm, globalObject, prototype);
    JSInjectedScriptHost* injectedScriptHost = JSInjectedScriptHost::create(vm, structure, Ref { *this });
    m_wrappers.addWrapper(globalObject, injectedScriptHost);

    return injectedScriptHost;
}

void InjectedScriptHost::clearAllWrappers()
{
    m_wrappers.clearAllWrappers();
}

} // namespace Inspector
