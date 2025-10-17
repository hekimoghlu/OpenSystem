/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 10, 2022.
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
#pragma once

#include "InspectorAgentBase.h"
#include "InspectorBackendDispatchers.h"
#include "JSCInlines.h"
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>

namespace JSC {
class Debugger;
}

namespace Inspector {

class InjectedScript;
class InjectedScriptManager;

class JS_EXPORT_PRIVATE InspectorAuditAgent : public InspectorAgentBase, public AuditBackendDispatcherHandler {
    WTF_MAKE_NONCOPYABLE(InspectorAuditAgent);
    WTF_MAKE_TZONE_ALLOCATED(InspectorAuditAgent);
public:
    ~InspectorAuditAgent() override;

    // InspectorAgentBase
    void didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*) final;
    void willDestroyFrontendAndBackend(DisconnectReason) final;

    // AuditBackendDispatcherHandler
    Protocol::ErrorStringOr<void> setup(std::optional<Protocol::Runtime::ExecutionContextId>&&) final;
    Protocol::ErrorStringOr<std::tuple<Ref<Protocol::Runtime::RemoteObject>, std::optional<bool> /* wasThrown */>> run(const String& test, std::optional<Protocol::Runtime::ExecutionContextId>&&) final;
    Protocol::ErrorStringOr<void> teardown() final;

    bool hasActiveAudit() const;

protected:
    InspectorAuditAgent(AgentContext&);

    InjectedScriptManager& injectedScriptManager() { return m_injectedScriptManager; }

    virtual InjectedScript injectedScriptForEval(Protocol::ErrorString&, std::optional<Protocol::Runtime::ExecutionContextId>&&) = 0;

    virtual void populateAuditObject(JSC::JSGlobalObject*, JSC::Strong<JSC::JSObject>& auditObject);

    virtual void muteConsole() { };
    virtual void unmuteConsole() { };

private:
    RefPtr<AuditBackendDispatcher> m_backendDispatcher;
    InjectedScriptManager& m_injectedScriptManager;
    JSC::Debugger& m_debugger;

    JSC::Strong<JSC::JSObject> m_injectedWebInspectorAuditValue;
};

} // namespace Inspector
