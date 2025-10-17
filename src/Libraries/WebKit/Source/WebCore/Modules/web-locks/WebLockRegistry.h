/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 25, 2022.
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

#include "ClientOrigin.h"
#include "ProcessIdentifier.h"
#include "ScriptExecutionContextIdentifier.h"
#include "WebLockIdentifier.h"
#include "WebLockMode.h"
#include <pal/SessionID.h>
#include <wtf/Deque.h>
#include <wtf/HashMap.h>
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class Exception;
struct WebLockManagerSnapshot;

class WebLockRegistry : public RefCounted<WebLockRegistry> {
public:
    static WebLockRegistry& shared();
    WEBCORE_EXPORT static void setSharedRegistry(Ref<WebLockRegistry>&&);

    virtual ~WebLockRegistry() { }

    virtual void requestLock(PAL::SessionID, const ClientOrigin&, WebLockIdentifier, ScriptExecutionContextIdentifier, const String& name, WebLockMode, bool steal, bool ifAvailable, Function<void(bool)>&& grantedHandler, Function<void()>&& lockStolenHandler) = 0;
    virtual void releaseLock(PAL::SessionID, const ClientOrigin&, WebLockIdentifier, ScriptExecutionContextIdentifier, const String& name) = 0;
    virtual void abortLockRequest(PAL::SessionID, const ClientOrigin&, WebLockIdentifier, ScriptExecutionContextIdentifier, const String& name, CompletionHandler<void(bool)>&&) = 0;
    virtual void snapshot(PAL::SessionID, const ClientOrigin&, CompletionHandler<void(WebLockManagerSnapshot&&)>&&) = 0;
    virtual void clientIsGoingAway(PAL::SessionID, const ClientOrigin&, ScriptExecutionContextIdentifier) = 0;

protected:
    WebLockRegistry() = default;
};

class LocalWebLockRegistry final : public WebLockRegistry, public CanMakeWeakPtr<LocalWebLockRegistry> {
public:
    static Ref<LocalWebLockRegistry> create() { return adoptRef(*new LocalWebLockRegistry); }
    ~LocalWebLockRegistry();

    WEBCORE_EXPORT void requestLock(PAL::SessionID, const ClientOrigin&, WebLockIdentifier, ScriptExecutionContextIdentifier, const String& name, WebLockMode, bool steal, bool ifAvailable, Function<void(bool)>&& grantedHandler, Function<void()>&& lockStolenHandler) final;
    WEBCORE_EXPORT void releaseLock(PAL::SessionID, const ClientOrigin&, WebLockIdentifier, ScriptExecutionContextIdentifier, const String& name) final;
    WEBCORE_EXPORT void abortLockRequest(PAL::SessionID, const ClientOrigin&, WebLockIdentifier, ScriptExecutionContextIdentifier, const String& name, CompletionHandler<void(bool)>&&) final;
    WEBCORE_EXPORT void snapshot(PAL::SessionID, const ClientOrigin&, CompletionHandler<void(WebLockManagerSnapshot&&)>&&) final;
    WEBCORE_EXPORT void clientIsGoingAway(PAL::SessionID, const ClientOrigin&, ScriptExecutionContextIdentifier) final;
    WEBCORE_EXPORT void clientsAreGoingAway(ProcessIdentifier);

private:
    WEBCORE_EXPORT LocalWebLockRegistry();

    class PerOriginRegistry;
    Ref<PerOriginRegistry> ensureRegistryForOrigin(PAL::SessionID, const ClientOrigin&);
    RefPtr<PerOriginRegistry> existingRegistryForOrigin(PAL::SessionID, const ClientOrigin&) const;

    HashMap<std::pair<PAL::SessionID, ClientOrigin>, WeakPtr<PerOriginRegistry>> m_perOriginRegistries;
};

} // namespace WebCore
