/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 28, 2025.
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

#include "MessageSender.h"
#include <WebCore/FrameIdentifier.h>
#include <WebCore/PageIdentifier.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class IntPoint;
}

namespace WebKit {

class WebPageProxy;

class WebPageProxyTesting : public IPC::MessageSender, public RefCounted<WebPageProxyTesting> {
    WTF_MAKE_TZONE_ALLOCATED(WebPageProxyTesting);
    WTF_MAKE_NONCOPYABLE(WebPageProxyTesting);
public:
    static Ref<WebPageProxyTesting> create(WebPageProxy& page) { return adoptRef(*new WebPageProxyTesting(page)); }

    void isLayerTreeFrozen(CompletionHandler<void(bool)>&&);
    void dispatchActivityStateUpdate();
    void setCrossSiteLoadWithLinkDecorationForTesting(const URL& fromURL, const URL& toURL, bool wasFiltered, CompletionHandler<void()>&&);
    void setPermissionLevel(const String& origin, bool allowed);
    bool isEditingCommandEnabled(const String& commandName);
    void resetStateBetweenTests();

    void dumpPrivateClickMeasurement(CompletionHandler<void(const String&)>&&);
    void clearPrivateClickMeasurement(CompletionHandler<void()>&&);
    void setPrivateClickMeasurementOverrideTimer(bool value, CompletionHandler<void()>&&);
    void markAttributedPrivateClickMeasurementsAsExpired(CompletionHandler<void()>&&);
    void setPrivateClickMeasurementEphemeralMeasurement(bool value, CompletionHandler<void()>&&);
    void simulatePrivateClickMeasurementSessionRestart(CompletionHandler<void()>&&);
    void setPrivateClickMeasurementTokenPublicKeyURL(const URL&, CompletionHandler<void()>&&);
    void setPrivateClickMeasurementTokenSignatureURL(const URL&, CompletionHandler<void()>&&);
    void setPrivateClickMeasurementAttributionReportURLs(const URL& sourceURL, const URL& destinationURL, CompletionHandler<void()>&&);
    void markPrivateClickMeasurementsAsExpired(CompletionHandler<void()>&&);
    void setPCMFraudPreventionValues(const String& unlinkableToken, const String& secretToken, const String& signature, const String& keyID, CompletionHandler<void()>&&);
    void setPrivateClickMeasurementAppBundleID(const String& appBundleIDForTesting, CompletionHandler<void()>&&);

#if ENABLE(NOTIFICATIONS)
    void clearNotificationPermissionState();
#endif

    void clearWheelEventTestMonitor();

#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)
    void setIndexOfGetDisplayMediaDeviceSelectedForTesting(std::optional<unsigned>);
    void setSystemCanPromptForGetDisplayMediaForTesting(bool);
#endif

    void setTopContentInset(float, CompletionHandler<void()>&&);

    void clearBackForwardList(CompletionHandler<void()>&&);

    void setTracksRepaints(bool, CompletionHandler<void()>&&);
    void displayAndTrackRepaints(CompletionHandler<void()>&&);

private:
    explicit WebPageProxyTesting(WebPageProxy&);

    bool sendMessage(UniqueRef<IPC::Encoder>&&, OptionSet<IPC::SendOption>) final;
    bool sendMessageWithAsyncReply(UniqueRef<IPC::Encoder>&&, AsyncReplyHandler, OptionSet<IPC::SendOption>) final;

    IPC::Connection* messageSenderConnection() const final;
    uint64_t messageSenderDestinationID() const final;

    Ref<WebPageProxy> protectedPage() const;

    WeakRef<WebPageProxy> m_page;
};

} // namespace WebKit
