/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 20, 2024.
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
#include <WebCore/ProcessQualified.h>
#include <WebCore/ScriptExecutionContextIdentifier.h>
#include <WebCore/WebLockRegistry.h>
#include <wtf/HashMap.h>

namespace WebKit {

class WebProcess;

class RemoteWebLockRegistry final : public WebCore::WebLockRegistry, public IPC::MessageReceiver {
public:
    static Ref<RemoteWebLockRegistry> create(WebProcess& process) { return adoptRef(*new RemoteWebLockRegistry(process)); }
    ~RemoteWebLockRegistry();

    void ref() const final { WebCore::WebLockRegistry::ref(); }
    void deref() const final { WebCore::WebLockRegistry::deref(); }

    // WebCore::WebLockRegistry.
    void requestLock(PAL::SessionID, const WebCore::ClientOrigin&, WebCore::WebLockIdentifier, WebCore::ScriptExecutionContextIdentifier, const String& name, WebCore::WebLockMode, bool steal, bool ifAvailable, Function<void(bool)>&& grantedHandler, Function<void()>&& lockStolenHandler) final;
    void releaseLock(PAL::SessionID, const WebCore::ClientOrigin&, WebCore::WebLockIdentifier, WebCore::ScriptExecutionContextIdentifier, const String& name) final;
    void abortLockRequest(PAL::SessionID, const WebCore::ClientOrigin&, WebCore::WebLockIdentifier, WebCore::ScriptExecutionContextIdentifier, const String& name, CompletionHandler<void(bool)>&&) final;
    void snapshot(PAL::SessionID, const WebCore::ClientOrigin&, CompletionHandler<void(WebCore::WebLockManagerSnapshot&&)>&&) final;
    void clientIsGoingAway(PAL::SessionID, const WebCore::ClientOrigin&, WebCore::ScriptExecutionContextIdentifier) final;

    // IPC::MessageReceiver.
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

private:
    explicit RemoteWebLockRegistry(WebProcess&);

    // IPC Message handlers.
    void didCompleteLockRequest(WebCore::WebLockIdentifier, WebCore::ScriptExecutionContextIdentifier, bool success);
    void didStealLock(WebCore::WebLockIdentifier, WebCore::ScriptExecutionContextIdentifier);

    struct LocksSnapshot;
    HashMap<WebCore::ScriptExecutionContextIdentifier, LocksSnapshot> m_locksSnapshotPerClient;
};

} // namespace WebKit
