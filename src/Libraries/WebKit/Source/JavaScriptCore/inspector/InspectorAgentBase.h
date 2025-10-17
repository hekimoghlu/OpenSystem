/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 21, 2022.
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

#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace JSC {
class JSGlobalObject;
}

namespace Inspector {

class BackendDispatcher;
class FrontendRouter;
class InjectedScriptManager;
class InspectorEnvironment;

struct AgentContext {
    InspectorEnvironment& environment;
    InjectedScriptManager& injectedScriptManager;
    FrontendRouter& frontendRouter;
    BackendDispatcher& backendDispatcher;
};

struct JSAgentContext : public AgentContext {
    JSAgentContext(AgentContext& context, JSC::JSGlobalObject& globalObject)
        : AgentContext(context)
        , inspectedGlobalObject(globalObject)
    {
    }

    JSC::JSGlobalObject& inspectedGlobalObject;
};

enum class DisconnectReason {
    InspectedTargetDestroyed,
    InspectorDestroyed
};

class InspectorAgentBase {
    WTF_MAKE_TZONE_ALLOCATED(InspectorAgentBase);
public:
    virtual ~InspectorAgentBase() { }

    String domainName() const { return m_name; }

    virtual void didCreateFrontendAndBackend(FrontendRouter*, BackendDispatcher*) = 0;
    virtual void willDestroyFrontendAndBackend(DisconnectReason) = 0;
    virtual void discardValues() { }
    virtual void discardAgent() { }

protected:
    InspectorAgentBase(const String& name)
        : m_name(name)
    {
    }

    String m_name;
};

} // namespace Inspector
