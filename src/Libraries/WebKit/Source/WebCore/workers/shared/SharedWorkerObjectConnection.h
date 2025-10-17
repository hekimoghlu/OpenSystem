/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 28, 2024.
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

#include "SharedWorkerObjectIdentifier.h"
#include "TransferredMessagePort.h"
#include <wtf/Forward.h>
#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class MessagePort;
class ResourceError;
class SharedWorkerScriptLoader;

struct SharedWorkerKey;
struct WorkerFetchResult;
struct WorkerInitializationData;
struct WorkerOptions;

class SharedWorkerObjectConnection : public RefCounted<SharedWorkerObjectConnection> {
public:
    WEBCORE_EXPORT virtual ~SharedWorkerObjectConnection();

    virtual void requestSharedWorker(const SharedWorkerKey&, SharedWorkerObjectIdentifier, TransferredMessagePort&&, const WorkerOptions&) = 0;
    virtual void sharedWorkerObjectIsGoingAway(const SharedWorkerKey&, SharedWorkerObjectIdentifier) = 0;
    virtual void suspendForBackForwardCache(const SharedWorkerKey&, SharedWorkerObjectIdentifier) = 0;
    virtual void resumeForBackForwardCache(const SharedWorkerKey&, SharedWorkerObjectIdentifier) = 0;

protected:
    // IPC messages.
    WEBCORE_EXPORT void fetchScriptInClient(URL&&, WebCore::SharedWorkerObjectIdentifier, WorkerOptions&&, CompletionHandler<void(WorkerFetchResult&&, WorkerInitializationData&&)>&&);
    WEBCORE_EXPORT void notifyWorkerObjectOfLoadCompletion(WebCore::SharedWorkerObjectIdentifier, const ResourceError&);
    WEBCORE_EXPORT void postErrorToWorkerObject(SharedWorkerObjectIdentifier, const String& errorMessage, int lineNumber, int columnNumber, const String& sourceURL, bool isErrorEvent);

    WEBCORE_EXPORT SharedWorkerObjectConnection();

private:
    HashMap<uint64_t, UniqueRef<SharedWorkerScriptLoader>> m_loaders;
};

} // namespace WebCore
