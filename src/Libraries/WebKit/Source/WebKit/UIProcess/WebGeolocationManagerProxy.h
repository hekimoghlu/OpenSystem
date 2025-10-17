/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 7, 2022.
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

#include "APIObject.h"
#include "Connection.h"
#include "MessageReceiver.h"
#include "WebContextSupplement.h"
#include "WebPageProxyIdentifier.h"
#include <WebCore/GeolocationPositionData.h>
#include <WebCore/RegistrableDomain.h>
#include <wtf/HashMap.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(IOS_FAMILY)
#include <WebCore/CoreLocationGeolocationProvider.h>
#endif

namespace API {
class GeolocationProvider;
}

namespace WebKit {

class WebGeolocationPosition;
class WebProcessPool;

class WebGeolocationManagerProxy : public API::ObjectImpl<API::Object::Type::GeolocationManager>, public WebContextSupplement, private IPC::MessageReceiver
#if PLATFORM(IOS_FAMILY)
    , public WebCore::CoreLocationGeolocationProvider::Client
#endif
{
public:
    static ASCIILiteral supplementName();

    static Ref<WebGeolocationManagerProxy> create(WebProcessPool*);
    ~WebGeolocationManagerProxy();

    void ref() const final { API::ObjectImpl<API::Object::Type::GeolocationManager>::ref(); }
    void deref() const final { API::ObjectImpl<API::Object::Type::GeolocationManager>::deref(); }

    void setProvider(std::unique_ptr<API::GeolocationProvider>&&);

    void providerDidChangePosition(WebGeolocationPosition*);
    void providerDidFailToDeterminePosition(const String& errorMessage = String());
#if PLATFORM(IOS_FAMILY)
    void resetPermissions();
#endif

    using API::Object::ref;
    using API::Object::deref;

    void webProcessIsGoingAway(WebProcessProxy&);
    std::optional<SharedPreferencesForWebProcess> sharedPreferencesForWebProcess(IPC::Connection&) const;

private:
    explicit WebGeolocationManagerProxy(WebProcessPool*);

    // WebContextSupplement
    void processPoolDestroyed() override;
    void refWebContextSupplement() override;
    void derefWebContextSupplement() override;

    // IPC::MessageReceiver
    void didReceiveMessage(IPC::Connection&, IPC::Decoder&) override;

    // IPC messages.
    void startUpdating(IPC::Connection&, const WebCore::RegistrableDomain&, WebPageProxyIdentifier, const String& authorizationToken, bool enableHighAccuracy);
    void stopUpdating(IPC::Connection&, const WebCore::RegistrableDomain&);
    void setEnableHighAccuracy(IPC::Connection&, const WebCore::RegistrableDomain&, bool);

    void startUpdatingWithProxy(WebProcessProxy&, const WebCore::RegistrableDomain&, WebPageProxyIdentifier, const String& authorizationToken, bool enableHighAccuracy);
    void stopUpdatingWithProxy(WebProcessProxy&, const WebCore::RegistrableDomain&);
    void setEnableHighAccuracyWithProxy(WebProcessProxy&, const WebCore::RegistrableDomain&, bool);

#if PLATFORM(IOS_FAMILY)
    // CoreLocationGeolocationProvider::Client
    void positionChanged(const String& websiteIdentifier, WebCore::GeolocationPositionData&&) final;
    void errorOccurred(const String& websiteIdentifier, const String& errorMessage) final;
    void resetGeolocation(const String& websiteIdentifier) final;
#endif

    struct PerDomainData {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;
        WeakHashSet<WebProcessProxy> watchers;
        WeakHashSet<WebProcessProxy> watchersNeedingHighAccuracy;
        std::optional<WebCore::GeolocationPositionData> lastPosition;

        // FIXME: Use for all Cocoa ports.
#if PLATFORM(IOS_FAMILY)
        std::unique_ptr<WebCore::CoreLocationGeolocationProvider> provider;
#endif
    };

    bool isUpdating(const PerDomainData&) const;
    bool isHighAccuracyEnabled(const PerDomainData&) const;
    void providerStartUpdating(PerDomainData&, const WebCore::RegistrableDomain&);
    void providerStopUpdating(PerDomainData&);
    void providerSetEnabledHighAccuracy(PerDomainData&, bool enabled);

    HashMap<WebCore::RegistrableDomain, std::unique_ptr<PerDomainData>> m_perDomainData;
    std::unique_ptr<API::GeolocationProvider> m_clientProvider;
};

} // namespace WebKit
