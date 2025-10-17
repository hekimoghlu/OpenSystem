/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 13, 2023.
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

#include "InspectorCanvasAgent.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class WorkerOrWorkletGlobalScope;

class WorkerCanvasAgent final : public InspectorCanvasAgent {
    WTF_MAKE_NONCOPYABLE(WorkerCanvasAgent);
    WTF_MAKE_TZONE_ALLOCATED(WorkerCanvasAgent);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WorkerCanvasAgent);
public:
    WorkerCanvasAgent(WorkerAgentContext&);
    ~WorkerCanvasAgent();

    // CanvasBackendDispatcherHandler
    Inspector::Protocol::ErrorStringOr<Inspector::Protocol::DOM::NodeId> requestNode(const Inspector::Protocol::Canvas::CanvasId&) override;
    Inspector::Protocol::ErrorStringOr<Ref<JSON::ArrayOf<Inspector::Protocol::DOM::NodeId>>> requestClientNodes(const Inspector::Protocol::Canvas::CanvasId&) override;

private:
    bool matchesCurrentContext(ScriptExecutionContext*) const override;

    WorkerOrWorkletGlobalScope& m_globalScope;
};

} // namespace WebCore
