/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 18, 2024.
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

#include "AuthenticatorTransportService.h"
#include "VirtualAuthenticatorConfiguration.h"
#include "VirtualCredential.h"
#include <wtf/RefCounted.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class VirtualAuthenticatorManager;

class VirtualService final: public AuthenticatorTransportService, public RefCounted<VirtualService> {
    WTF_MAKE_TZONE_ALLOCATED(VirtualService);
public:
    static Ref<VirtualService> create(AuthenticatorTransportServiceObserver&, Vector<std::pair<String, VirtualAuthenticatorConfiguration>>&);

    static Ref<AuthenticatorTransportService> createVirtual(WebCore::AuthenticatorTransport, AuthenticatorTransportServiceObserver&, Vector<std::pair<String, VirtualAuthenticatorConfiguration>>&);

    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

private:
    explicit VirtualService(AuthenticatorTransportServiceObserver&, Vector<std::pair<String, VirtualAuthenticatorConfiguration>>&);

    void startDiscoveryInternal() final;

    Vector<std::pair<String, VirtualAuthenticatorConfiguration>> m_authenticators;
};

} // namespace WebKit

#endif // ENABLE(WEB_AUTHN)
