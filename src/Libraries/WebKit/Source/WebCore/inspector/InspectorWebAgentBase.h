/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 26, 2022.
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

#include "Page.h"
#include "WorkerOrWorkletGlobalScope.h"
#include <JavaScriptCore/InspectorAgentBase.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class InstrumentingAgents;

// FIXME: move this to Inspector namespace when remaining agents move.
struct WebAgentContext : public Inspector::AgentContext {
    WebAgentContext(AgentContext& context, InstrumentingAgents& instrumentingAgents)
        : AgentContext(context)
        , instrumentingAgents(instrumentingAgents)
    {
    }

    InstrumentingAgents& instrumentingAgents;
};

struct PageAgentContext : public WebAgentContext {
    PageAgentContext(WebAgentContext& context, Page& inspectedPage)
        : WebAgentContext(context)
        , inspectedPage(inspectedPage)
    {
    }

    WeakRef<Page> inspectedPage;
};

struct WorkerAgentContext : public WebAgentContext {
    WorkerAgentContext(WebAgentContext& context, WorkerOrWorkletGlobalScope& globalScope)
        : WebAgentContext(context)
        , globalScope(globalScope)
    {
    }

    WeakRef<WorkerOrWorkletGlobalScope> globalScope;
};

class InspectorAgentBase : public Inspector::InspectorAgentBase {
protected:
    InspectorAgentBase(const String& name, WebAgentContext& context)
        : Inspector::InspectorAgentBase(name)
        , m_instrumentingAgents(context.instrumentingAgents)
        , m_environment(context.environment)
    {
    }

    InstrumentingAgents& m_instrumentingAgents;
    Inspector::InspectorEnvironment& m_environment;
};
    
} // namespace WebCore
