/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 18, 2023.
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
#include "CommandLineAPIModule.h"

#include "JSDOMGlobalObject.h"
#include "WebCoreJSBuiltinInternals.h"
#include "WebInjectedScriptManager.h"
#include <JavaScriptCore/HeapInlines.h>
#include <JavaScriptCore/InjectedScript.h>

namespace WebCore {

using namespace JSC;
using namespace Inspector;

void CommandLineAPIModule::injectIfNeeded(InjectedScriptManager* injectedScriptManager, const InjectedScript& injectedScript)
{
    CommandLineAPIModule module;
    module.ensureInjected(injectedScriptManager, injectedScript);
}

CommandLineAPIModule::CommandLineAPIModule()
    : InjectedScriptModule("CommandLineAPI"_s)
{
}

JSFunction* CommandLineAPIModule::injectModuleFunction(JSC::JSGlobalObject* lexicalGlobalObject) const
{
    if (auto* globalObject = jsCast<JSDOMGlobalObject*>(lexicalGlobalObject))
        return globalObject->builtinInternalFunctions().commandLineAPIModuleSource().m_injectModuleFunction.get();

    WTFLogAlways("Attempted to get `injectModule` function from `CommandLineAPIModule` for non-`JSDOMGlobalObject`.");
    RELEASE_ASSERT_NOT_REACHED();
    return nullptr;
}

JSValue CommandLineAPIModule::host(InjectedScriptManager* injectedScriptManager, JSGlobalObject* lexicalGlobalObject) const
{
    // CommandLineAPIModule should only ever be used by a WebInjectedScriptManager.
    WebInjectedScriptManager* pageInjectedScriptManager = static_cast<WebInjectedScriptManager*>(injectedScriptManager);
    ASSERT(pageInjectedScriptManager->commandLineAPIHost());

    JSDOMGlobalObject* globalObject = jsCast<JSDOMGlobalObject*>(lexicalGlobalObject);
    return pageInjectedScriptManager->commandLineAPIHost()->wrapper(lexicalGlobalObject, globalObject);
}

} // namespace WebCore
