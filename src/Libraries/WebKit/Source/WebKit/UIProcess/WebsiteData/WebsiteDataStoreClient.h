/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 16, 2024.
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

#include "AuthenticationChallengeDisposition.h"
#include "AuthenticationChallengeProxy.h"
#include "AuthenticationDecisionListener.h"
#include "BackgroundFetchChange.h"
#include <WebCore/NotificationData.h>
#include <wtf/CompletionHandler.h>
#include <wtf/Seconds.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {
enum class WasPrivateRelayed : bool;
enum class WindowProxyProperty : uint8_t;
struct NotificationData;
class RegistrableDomain;
class SecurityOriginData;
}

namespace WebKit {

class WebPageProxy;
class WebsiteDataStore;

class WebsiteDataStoreClient {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WebsiteDataStoreClient);
public:
    virtual ~WebsiteDataStoreClient() { }

    virtual void requestStorageSpace(const WebCore::SecurityOriginData& topOrigin, const WebCore::SecurityOriginData& frameOrigin, uint64_t quota, uint64_t currentSize, uint64_t spaceRequired, CompletionHandler<void(std::optional<uint64_t>)>&& completionHandler)
    {
        UNUSED_PARAM(topOrigin);
        UNUSED_PARAM(frameOrigin);
        UNUSED_PARAM(quota);
        UNUSED_PARAM(currentSize);
        UNUSED_PARAM(spaceRequired);
        completionHandler({ });
    }

    virtual void didReceiveAuthenticationChallenge(Ref<AuthenticationChallengeProxy>&& challenge)
    {
        challenge->listener().completeChallenge(AuthenticationChallengeDisposition::PerformDefaultHandling);
    }

    virtual void openWindowFromServiceWorker(const String&, const WebCore::SecurityOriginData&, CompletionHandler<void(WebPageProxy*)>&& completionHandler)
    {
        completionHandler(nullptr);
    }
    virtual void reportServiceWorkerConsoleMessage(const URL&, const WebCore::SecurityOriginData&, MessageSource, MessageLevel, const String&, unsigned long)
    {
    }

    virtual bool showNotification(const WebCore::NotificationData&)
    {
        return false;
    }

    virtual HashMap<WTF::String, bool> notificationPermissions()
    {
        return { };
    }

    virtual bool hasGetDisplayedNotifications() const { return false; }

    virtual void getDisplayedNotifications(const WebCore::SecurityOriginData&, CompletionHandler<void(Vector<WebCore::NotificationData>&&)>&& completionHandler)
    {
        completionHandler({ });
    }

    virtual void workerUpdatedAppBadge(const WebCore::SecurityOriginData&, std::optional<uint64_t>)
    {
    }

    virtual void navigationToNotificationActionURL(const URL&)
    {
    }

    virtual void requestBackgroundFetchPermission(const WebCore::SecurityOriginData& topOrigin, const WebCore::SecurityOriginData& frameOrigin, CompletionHandler<void(bool)>&& completionHandler)
    {
        UNUSED_PARAM(topOrigin);
        UNUSED_PARAM(frameOrigin);
        completionHandler(false);
    }

    virtual void notifyBackgroundFetchChange(const String&, BackgroundFetchChange)
    {
    }

    virtual void didAccessWindowProxyProperty(const WebCore::RegistrableDomain&, const WebCore::RegistrableDomain&, WebCore::WindowProxyProperty, bool)
    {
    }

    virtual void didAllowPrivateTokenUsageByThirdPartyForTesting(bool, URL&&)
    {
    }

    enum class CanSuspend : bool { No, Yes };
    virtual void didExceedMemoryFootprintThreshold(size_t, const String&, unsigned, Seconds, bool, WebCore::WasPrivateRelayed, CanSuspend)
    {
    }
    virtual void webCryptoMasterKey(CompletionHandler<void(std::optional<Vector<uint8_t>>&&)>&& completionHandler)
    {
        return completionHandler(std::nullopt);
    }
};

} // namespace WebKit
