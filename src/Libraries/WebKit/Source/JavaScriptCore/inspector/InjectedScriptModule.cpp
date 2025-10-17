/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 10, 2024.
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
#include "InjectedScriptModule.h"

#include "InjectedScript.h"
#include "InjectedScriptManager.h"
#include "ScriptFunctionCall.h"

namespace Inspector {

InjectedScriptModule::InjectedScriptModule(const String& name)
    : InjectedScriptBase(name)
{
}

InjectedScriptModule::~InjectedScriptModule() = default;

void InjectedScriptModule::ensureInjected(InjectedScriptManager* injectedScriptManager, JSC::JSGlobalObject* globalObject)
{
    InjectedScript injectedScript = injectedScriptManager->injectedScriptFor(globalObject);
    ensureInjected(injectedScriptManager, injectedScript);
}

void InjectedScriptModule::ensureInjected(InjectedScriptManager* injectedScriptManager, const InjectedScript& injectedScript)
{
    ASSERT(!injectedScript.hasNoValue());
    if (injectedScript.hasNoValue())
        return;

    // FIXME: Make the InjectedScript a module itself.
    JSC::JSLockHolder locker(injectedScript.globalObject());
    ScriptFunctionCall function(injectedScript.globalObject(), injectedScript.injectedScriptObject(), "hasInjectedModule"_s, injectedScriptManager->inspectorEnvironment().functionCallHandler());
    function.appendArgument(name());
    auto hasInjectedModuleResult = injectedScript.callFunctionWithEvalEnabled(function);
    ASSERT(hasInjectedModuleResult);
    if (!hasInjectedModuleResult) {
        auto& error = hasInjectedModuleResult.error();
        ASSERT(error);
        JSC::LineColumn lineColumn;
        auto& stack = error->stack();
        if (stack.size() > 0)
            lineColumn = stack[0].computeLineAndColumn();
        WTFLogAlways("Error when calling 'hasInjectedModule' for '%s': %s (%d:%d)\n", name().utf8().data(), error->value().toWTFString(injectedScript.globalObject()).utf8().data(), lineColumn.line, lineColumn.column);
        RELEASE_ASSERT_NOT_REACHED();
    }
    if (!hasInjectedModuleResult.value()) {
        WTFLogAlways("VM is terminated when calling 'injectModule' for '%s'\n", name().utf8().data());
        RELEASE_ASSERT_NOT_REACHED();
    }
    if (!hasInjectedModuleResult.value().isBoolean() || !hasInjectedModuleResult.value().asBoolean()) {
        ScriptFunctionCall function(injectedScript.globalObject(), injectedScript.injectedScriptObject(), "injectModule"_s, injectedScriptManager->inspectorEnvironment().functionCallHandler());
        function.appendArgument(name());
        function.appendArgument(JSC::JSValue(injectModuleFunction(injectedScript.globalObject())));
        function.appendArgument(host(injectedScriptManager, injectedScript.globalObject()));
        auto injectModuleResult = injectedScript.callFunctionWithEvalEnabled(function);
        if (!injectModuleResult) {
            auto& error = injectModuleResult.error();
            ASSERT(error);
            JSC::LineColumn lineColumn;
            auto& stack = error->stack();
            if (stack.size() > 0)
                lineColumn = stack[0].computeLineAndColumn();
            WTFLogAlways("Error when calling 'injectModule' for '%s': %s (%d:%d)\n", name().utf8().data(), error->value().toWTFString(injectedScript.globalObject()).utf8().data(), lineColumn.line, lineColumn.column);
            RELEASE_ASSERT_NOT_REACHED();
        }
    }
}

} // namespace Inspector
