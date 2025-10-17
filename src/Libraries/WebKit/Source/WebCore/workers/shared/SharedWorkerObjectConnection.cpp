/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 27, 2022.
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
#include "SharedWorkerObjectConnection.h"

#include "ActiveDOMObject.h"
#include "ErrorEvent.h"
#include "EventNames.h"
#include "Logging.h"
#include "ScriptBuffer.h"
#include "SharedWorker.h"
#include "SharedWorkerScriptLoader.h"
#include "WorkerFetchResult.h"
#include "WorkerInitializationData.h"
#include "WorkerScriptLoader.h"

namespace WebCore {

#define CONNECTION_RELEASE_LOG(fmt, ...) RELEASE_LOG(SharedWorker, "%p - SharedWorkerObjectConnection::" fmt, this, ##__VA_ARGS__)
#define CONNECTION_RELEASE_LOG_ERROR(fmt, ...) RELEASE_LOG_ERROR(SharedWorker, "%p - SharedWorkerObjectConnection::" fmt, this, ##__VA_ARGS__)

static uint64_t loaderIdentifierSeed = 0;

SharedWorkerObjectConnection::SharedWorkerObjectConnection() = default;

SharedWorkerObjectConnection::~SharedWorkerObjectConnection() = default;

void SharedWorkerObjectConnection::fetchScriptInClient(URL&& url, WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier, WorkerOptions&& workerOptions, CompletionHandler<void(WorkerFetchResult&&, WorkerInitializationData&&)>&& completionHandler)
{
    ASSERT(isMainThread());

    auto* workerObject = SharedWorker::fromIdentifier(sharedWorkerObjectIdentifier);
    CONNECTION_RELEASE_LOG("fetchScriptInClient: sharedWorkerObjectIdentifier=%" PUBLIC_LOG_STRING ", worker=%p", sharedWorkerObjectIdentifier.toString().utf8().data(), workerObject);
    if (!workerObject)
        return completionHandler(workerFetchError(ResourceError { ResourceError::Type::Cancellation }), { });

    auto loaderIdentifier = ++loaderIdentifierSeed;
    auto loader = makeUniqueRef<SharedWorkerScriptLoader>(WTFMove(url), *workerObject, WTFMove(workerOptions));
    auto loaderPtr = loader.ptr();
    m_loaders.add(loaderIdentifier, WTFMove(loader));

    loaderPtr->load([this, loaderIdentifier, completionHandler = WTFMove(completionHandler)](WorkerFetchResult&& fetchResult, WorkerInitializationData&& initializationData) mutable {
        CONNECTION_RELEASE_LOG("fetchScriptInClient: finished script load, success=%d", fetchResult.error.isNull());
        auto loader = m_loaders.take(loaderIdentifier);
        ASSERT(loader);
        completionHandler(WTFMove(fetchResult), WTFMove(initializationData));
    });
}

void SharedWorkerObjectConnection::notifyWorkerObjectOfLoadCompletion(WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier, const ResourceError& error)
{
    ASSERT(isMainThread());
    auto* workerObject = SharedWorker::fromIdentifier(sharedWorkerObjectIdentifier);
    if (error.isNull())
        CONNECTION_RELEASE_LOG("notifyWorkerObjectOfLoadCompletion: sharedWorkerObjectIdentifier=%" PUBLIC_LOG_STRING ", worker=%p, load succeeded", sharedWorkerObjectIdentifier.toString().utf8().data(), workerObject);
    else
        CONNECTION_RELEASE_LOG_ERROR("notifyWorkerObjectOfLoadCompletion: sharedWorkerObjectIdentifier=%" PUBLIC_LOG_STRING ", worker=%p, load failed", sharedWorkerObjectIdentifier.toString().utf8().data(), workerObject);
    if (workerObject)
        workerObject->didFinishLoading(error);
}

void SharedWorkerObjectConnection::postErrorToWorkerObject(SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier, const String& errorMessage, int lineNumber, int columnNumber, const String& sourceURL, bool isErrorEvent)
{
    ASSERT(isMainThread());
    auto* workerObject = SharedWorker::fromIdentifier(sharedWorkerObjectIdentifier);
    CONNECTION_RELEASE_LOG_ERROR("postErrorToWorkerObject: sharedWorkerObjectIdentifier=%" PUBLIC_LOG_STRING ", worker=%p", sharedWorkerObjectIdentifier.toString().utf8().data(), workerObject);
    if (!workerObject)
        return;

    auto event = isErrorEvent ? Ref<Event> { ErrorEvent::create(errorMessage, sourceURL, lineNumber, columnNumber, { }) } : Event::create(eventNames().errorEvent, Event::CanBubble::No, Event::IsCancelable::No);
    ActiveDOMObject::queueTaskToDispatchEvent(*workerObject, TaskSource::DOMManipulation, WTFMove(event));
}

#undef CONNECTION_RELEASE_LOG
#undef CONNECTION_RELEASE_LOG_ERROR

} // namespace WebCore
