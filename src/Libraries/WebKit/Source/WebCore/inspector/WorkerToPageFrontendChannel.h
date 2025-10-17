/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 24, 2023.
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

#include "WorkerDebuggerProxy.h"
#include "WorkerOrWorkletGlobalScope.h"
#include "WorkerThread.h"
#include <JavaScriptCore/InspectorFrontendChannel.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

class WorkerToPageFrontendChannel final : public Inspector::FrontendChannel {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WorkerToPageFrontendChannel);
public:
    explicit WorkerToPageFrontendChannel(WorkerOrWorkletGlobalScope& globalScope)
        : m_globalScope(globalScope)
    {
    }
    ~WorkerToPageFrontendChannel() override = default;

private:
    ConnectionType connectionType() const override { return ConnectionType::Local; }

    void sendMessageToFrontend(const String& message) override
    {
        if (auto* workerDebuggerProxy = m_globalScope.workerOrWorkletThread()->workerDebuggerProxy())
            workerDebuggerProxy->postMessageToDebugger(message);
    }

    WorkerOrWorkletGlobalScope& m_globalScope;
};

} // namespace WebCore
