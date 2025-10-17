/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 23, 2023.
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

#if ENABLE(WEB_AUTHN)

#include "WebAuthenticationFlags.h"
#include <WebCore/AuthenticatorTransport.h>
#include <wtf/AbstractRefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
struct MockWebAuthenticationConfiguration;
}

namespace WebKit {

class Authenticator;

class AuthenticatorTransportServiceObserver : public AbstractRefCountedAndCanMakeWeakPtr<AuthenticatorTransportServiceObserver> {
public:
    virtual ~AuthenticatorTransportServiceObserver() = default;

    virtual void authenticatorAdded(Ref<Authenticator>&&) = 0;
    virtual void serviceStatusUpdated(WebAuthenticationStatus) = 0;

protected:
    AuthenticatorTransportServiceObserver() = default;
};

class AuthenticatorTransportService : public AbstractRefCountedAndCanMakeWeakPtr<AuthenticatorTransportService> {
    WTF_MAKE_TZONE_ALLOCATED(AuthenticatorTransportService);
    WTF_MAKE_NONCOPYABLE(AuthenticatorTransportService);
public:
    static Ref<AuthenticatorTransportService> create(WebCore::AuthenticatorTransport, AuthenticatorTransportServiceObserver&);
    static Ref<AuthenticatorTransportService> createMock(WebCore::AuthenticatorTransport, AuthenticatorTransportServiceObserver&, const WebCore::MockWebAuthenticationConfiguration&);

    virtual ~AuthenticatorTransportService() = default;

    // These operations are guaranteed to execute asynchronously.
    void startDiscovery();
    void restartDiscovery();

protected:
    explicit AuthenticatorTransportService(AuthenticatorTransportServiceObserver&);

    AuthenticatorTransportServiceObserver* observer() const { return m_observer.get(); }

private:
    virtual void startDiscoveryInternal() = 0;
    // NFC service's polling is one shot. It halts after the first tags are detected.
    // Therefore, a restart process is needed to resume polling after exceptions.
    virtual void restartDiscoveryInternal() { };

    WeakPtr<AuthenticatorTransportServiceObserver> m_observer;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
