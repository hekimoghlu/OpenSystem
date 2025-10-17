/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 4, 2025.
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

#include "MessageReceiver.h"
#include "MessageSender.h"
#include <WebCore/SharedWorkerObjectConnection.h>

namespace WebKit {

class WebSharedWorkerObjectConnection final : public WebCore::SharedWorkerObjectConnection, private IPC::MessageSender, public IPC::MessageReceiver {
public:
    static Ref<WebSharedWorkerObjectConnection> create() { return adoptRef(*new WebSharedWorkerObjectConnection); }
    ~WebSharedWorkerObjectConnection();

    void ref() const final { WebCore::SharedWorkerObjectConnection::ref(); }
    void deref() const final { WebCore::SharedWorkerObjectConnection::deref(); }

    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

private:
    WebSharedWorkerObjectConnection();

    // WebCore::SharedWorkerObjectConnection.
    void requestSharedWorker(const WebCore::SharedWorkerKey&, WebCore::SharedWorkerObjectIdentifier, WebCore::TransferredMessagePort&&, const WebCore::WorkerOptions&) final;
    void sharedWorkerObjectIsGoingAway(const WebCore::SharedWorkerKey&, WebCore::SharedWorkerObjectIdentifier) final;
    void suspendForBackForwardCache(const WebCore::SharedWorkerKey&, WebCore::SharedWorkerObjectIdentifier) final;
    void resumeForBackForwardCache(const WebCore::SharedWorkerKey&, WebCore::SharedWorkerObjectIdentifier) final;

    // IPC::MessageSender.
    IPC::Connection* messageSenderConnection() const final;
    uint64_t messageSenderDestinationID() const final { return 0; }
};

} // namespace WebKit
