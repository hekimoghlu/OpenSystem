/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, April 11, 2024.
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

#include "MessagePortIdentifier.h"
#include "ResourceLoaderIdentifier.h"
#include "ResourceResponse.h"
#include "ScriptExecutionContextIdentifier.h"
#include "WorkerOptions.h"
#include "WorkerScriptLoaderClient.h"
#include <wtf/CompletionHandler.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

struct ServiceWorkerRegistrationData;
class SharedWorker;
class WorkerScriptLoader;
struct WorkerFetchResult;
struct WorkerInitializationData;

class SharedWorkerScriptLoader : private WorkerScriptLoaderClient {
    WTF_MAKE_TZONE_ALLOCATED(SharedWorkerScriptLoader);
public:
    SharedWorkerScriptLoader(URL&&, SharedWorker&, WorkerOptions&&);

    void load(CompletionHandler<void(WorkerFetchResult&&, WorkerInitializationData&&)>&&);

    const URL& url() const { return m_url; }
    SharedWorker& worker() { return m_worker.get(); }
    const WorkerOptions& options() const { return m_options; }

private:
    void didReceiveResponse(ScriptExecutionContextIdentifier, std::optional<ResourceLoaderIdentifier>, const ResourceResponse&) final;
    void notifyFinished(std::optional<ScriptExecutionContextIdentifier>) final;

    const WorkerOptions m_options;
    const Ref<SharedWorker> m_worker;
    const Ref<WorkerScriptLoader> m_loader;
    const URL m_url;
    CompletionHandler<void(WorkerFetchResult&&, WorkerInitializationData&&)> m_completionHandler;
};

} // namespace WebCore
