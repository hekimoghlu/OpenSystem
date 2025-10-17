/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 2, 2022.
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
#include "WebSharedWorkerObjectConnection.h"

#include "Logging.h"
#include "MessageSenderInlines.h"
#include "NetworkProcessConnection.h"
#include "WebMessagePortChannelProvider.h"
#include "WebProcess.h"
#include "WebSharedWorkerServerConnectionMessages.h"
#include <WebCore/ProcessIdentifier.h>
#include <WebCore/SharedWorkerKey.h>
#include <WebCore/WorkerOptions.h>

namespace WebKit {

#define CONNECTION_RELEASE_LOG(fmt, ...) RELEASE_LOG(SharedWorker, "%p - [webProcessIdentifier=%" PRIu64 "] WebSharedWorkerObjectConnection::" fmt, this, WebCore::Process::identifier().toUInt64(), ##__VA_ARGS__)

WebSharedWorkerObjectConnection::WebSharedWorkerObjectConnection()
{
    CONNECTION_RELEASE_LOG("WebSharedWorkerObjectConnection:");
}

WebSharedWorkerObjectConnection::~WebSharedWorkerObjectConnection()
{
    CONNECTION_RELEASE_LOG("~WebSharedWorkerObjectConnection:");
}

IPC::Connection* WebSharedWorkerObjectConnection::messageSenderConnection() const
{
    return &WebProcess::singleton().ensureNetworkProcessConnection().connection();
}

void WebSharedWorkerObjectConnection::requestSharedWorker(const WebCore::SharedWorkerKey& sharedWorkerKey, WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier, WebCore::TransferredMessagePort&& port, const WebCore::WorkerOptions& workerOptions)
{
    CONNECTION_RELEASE_LOG("requestSharedWorker: sharedWorkerObjectIdentifier=%" PUBLIC_LOG_STRING, sharedWorkerObjectIdentifier.toString().utf8().data());
    WebMessagePortChannelProvider::singleton().messagePortSentToRemote(port.first);
    send(Messages::WebSharedWorkerServerConnection::RequestSharedWorker { sharedWorkerKey, sharedWorkerObjectIdentifier, WTFMove(port), workerOptions });
}

void WebSharedWorkerObjectConnection::sharedWorkerObjectIsGoingAway(const WebCore::SharedWorkerKey& sharedWorkerKey, WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier)
{
    CONNECTION_RELEASE_LOG("sharedWorkerObjectIsGoingAway: sharedWorkerObjectIdentifier=%" PUBLIC_LOG_STRING, sharedWorkerObjectIdentifier.toString().utf8().data());
    send(Messages::WebSharedWorkerServerConnection::SharedWorkerObjectIsGoingAway { sharedWorkerKey, sharedWorkerObjectIdentifier });
}

void WebSharedWorkerObjectConnection::suspendForBackForwardCache(const WebCore::SharedWorkerKey& sharedWorkerKey, WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier)
{
    CONNECTION_RELEASE_LOG("suspendForBackForwardCache: sharedWorkerObjectIdentifier=%" PUBLIC_LOG_STRING, sharedWorkerObjectIdentifier.toString().utf8().data());
    send(Messages::WebSharedWorkerServerConnection::SuspendForBackForwardCache { sharedWorkerKey, sharedWorkerObjectIdentifier });
}

void WebSharedWorkerObjectConnection::resumeForBackForwardCache(const WebCore::SharedWorkerKey& sharedWorkerKey, WebCore::SharedWorkerObjectIdentifier sharedWorkerObjectIdentifier)
{
    CONNECTION_RELEASE_LOG("resumeForBackForwardCache: sharedWorkerObjectIdentifier=%" PUBLIC_LOG_STRING, sharedWorkerObjectIdentifier.toString().utf8().data());
    send(Messages::WebSharedWorkerServerConnection::ResumeForBackForwardCache { sharedWorkerKey, sharedWorkerObjectIdentifier });
}

#undef CONNECTION_RELEASE_LOG

} // namespace WebKit
