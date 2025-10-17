/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 6, 2023.
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
#include "WebProcessProxy.h"
#include <WebCore/ScriptExecutionContextIdentifier.h>
#include <WebCore/WebLockIdentifier.h>
#include <WebCore/WebLockMode.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {
struct ClientOrigin;
struct WebLockManagerSnapshot;
}

namespace WebKit {

struct SharedPreferencesForWebProcess;

class WebLockRegistryProxy final : public IPC::MessageReceiver {
    WTF_MAKE_TZONE_ALLOCATED(WebLockRegistryProxy);
public:
    explicit WebLockRegistryProxy(WebProcessProxy&);
    ~WebLockRegistryProxy();

    void ref() const final { m_process->ref(); }
    void deref() const final { m_process->deref(); }

    // FIXME: Remove SUPPRESS_UNCOUNTED_ARG once https://github.com/llvm/llvm-project/pull/111198 lands.
    SUPPRESS_UNCOUNTED_ARG std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess() { return m_process->sharedPreferencesForWebProcess(); }

    void processDidExit();
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) final;

private:
    // IPC Message handlers.
    void requestLock(WebCore::ClientOrigin&&, WebCore::WebLockIdentifier, WebCore::ScriptExecutionContextIdentifier, String&& name, WebCore::WebLockMode, bool steal, bool ifAvailable);
    void releaseLock(WebCore::ClientOrigin&&, WebCore::WebLockIdentifier, WebCore::ScriptExecutionContextIdentifier, String&& name);
    void abortLockRequest(WebCore::ClientOrigin&&, WebCore::WebLockIdentifier, WebCore::ScriptExecutionContextIdentifier, String&& name, CompletionHandler<void(bool)>&&);
    void snapshot(WebCore::ClientOrigin&&, CompletionHandler<void(WebCore::WebLockManagerSnapshot&&)>&&);
    void clientIsGoingAway(WebCore::ClientOrigin&&, WebCore::ScriptExecutionContextIdentifier);

    Ref<WebProcessProxy> protectedProcess() const { return m_process.get(); }

    CheckedRef<WebProcessProxy> m_process;
    bool m_hasEverRequestedLocks { false };
};

} // namespace WebKit

