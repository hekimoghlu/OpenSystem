/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 6, 2023.
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

#include "InspectorWebAgentBase.h"
#include <JavaScriptCore/InspectorAuditAgent.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class Page;

class PageAuditAgent final : public Inspector::InspectorAuditAgent {
    WTF_MAKE_NONCOPYABLE(PageAuditAgent);
    WTF_MAKE_TZONE_ALLOCATED(PageAuditAgent);
public:
    PageAuditAgent(PageAgentContext&);
    ~PageAuditAgent();

    Page& inspectedPage() const { return m_inspectedPage; }

private:
    Inspector::InjectedScript injectedScriptForEval(std::optional<Inspector::Protocol::Runtime::ExecutionContextId>&&);
    Inspector::InjectedScript injectedScriptForEval(Inspector::Protocol::ErrorString&, std::optional<Inspector::Protocol::Runtime::ExecutionContextId>&&);

    void populateAuditObject(JSC::JSGlobalObject*, JSC::Strong<JSC::JSObject>& auditObject);

    void muteConsole();
    void unmuteConsole();

    WeakRef<Page> m_inspectedPage;
};

} // namespace WebCore
