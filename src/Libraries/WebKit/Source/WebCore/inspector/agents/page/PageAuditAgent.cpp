/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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
#include "PageAuditAgent.h"

#include "InspectorAuditAccessibilityObject.h"
#include "InspectorAuditDOMObject.h"
#include "InspectorAuditResourcesObject.h"
#include "JSDOMWindowCustom.h"
#include "JSInspectorAuditAccessibilityObject.h"
#include "JSInspectorAuditDOMObject.h"
#include "JSInspectorAuditResourcesObject.h"
#include "Page.h"
#include "PageConsoleClient.h"
#include <JavaScriptCore/CallFrame.h>
#include <JavaScriptCore/InjectedScript.h>
#include <JavaScriptCore/InjectedScriptManager.h>
#include <JavaScriptCore/JSLock.h>
#include <wtf/Ref.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/Vector.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

using namespace Inspector;

WTF_MAKE_TZONE_ALLOCATED_IMPL(PageAuditAgent);

PageAuditAgent::PageAuditAgent(PageAgentContext& context)
    : InspectorAuditAgent(context)
    , m_inspectedPage(context.inspectedPage)
{
}

PageAuditAgent::~PageAuditAgent() = default;

InjectedScript PageAuditAgent::injectedScriptForEval(std::optional<Inspector::Protocol::Runtime::ExecutionContextId>&& executionContextId)
{
    if (executionContextId)
        return injectedScriptManager().injectedScriptForId(*executionContextId);
    if (RefPtr localMainFrame = m_inspectedPage->localMainFrame())
        return injectedScriptManager().injectedScriptFor(&mainWorldGlobalObject(*localMainFrame));
    return InjectedScript();
}

InjectedScript PageAuditAgent::injectedScriptForEval(Inspector::Protocol::ErrorString& errorString, std::optional<Inspector::Protocol::Runtime::ExecutionContextId>&& executionContextId)
{
    InjectedScript injectedScript = injectedScriptForEval(WTFMove(executionContextId));
    if (injectedScript.hasNoValue()) {
        if (executionContextId)
            errorString = "Missing injected script for given executionContextId"_s;
        else
            errorString = "Internal error: main world execution context not found"_s;
    }
    return injectedScript;
}

void PageAuditAgent::populateAuditObject(JSC::JSGlobalObject* lexicalGlobalObject, JSC::Strong<JSC::JSObject>& auditObject)
{
    InspectorAuditAgent::populateAuditObject(lexicalGlobalObject, auditObject);

    ASSERT(lexicalGlobalObject);
    if (!lexicalGlobalObject)
        return;

    if (auto* globalObject = JSC::jsCast<JSDOMGlobalObject*>(lexicalGlobalObject)) {
        JSC::VM& vm = globalObject->vm();
        JSC::JSLockHolder lock(vm);

        if (JSC::JSValue jsInspectorAuditAccessibilityObject = toJSNewlyCreated(lexicalGlobalObject, globalObject, InspectorAuditAccessibilityObject::create(*this)))
            auditObject->putDirect(vm, JSC::Identifier::fromString(vm, "Accessibility"_s), jsInspectorAuditAccessibilityObject);

        if (JSC::JSValue jsInspectorAuditDOMObject = toJSNewlyCreated(lexicalGlobalObject, globalObject, InspectorAuditDOMObject::create(*this)))
            auditObject->putDirect(vm, JSC::Identifier::fromString(vm, "DOM"_s), jsInspectorAuditDOMObject);

        if (JSC::JSValue jsInspectorAuditResourcesObject = toJSNewlyCreated(lexicalGlobalObject, globalObject, InspectorAuditResourcesObject::create(*this)))
            auditObject->putDirect(vm, JSC::Identifier::fromString(vm, "Resources"_s), jsInspectorAuditResourcesObject);
    }
}

void PageAuditAgent::muteConsole()
{
    InspectorAuditAgent::muteConsole();
    PageConsoleClient::mute();
}

void PageAuditAgent::unmuteConsole()
{
    PageConsoleClient::unmute();
    InspectorAuditAgent::unmuteConsole();
}

} // namespace WebCore
