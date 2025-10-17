/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 16, 2024.
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

#include "PrivateClickMeasurementConnection.h"
#include "PrivateClickMeasurementManagerInterface.h"
#include <wtf/TZoneMalloc.h>
#include <wtf/text/CString.h>

namespace WebKit {

class NetworkSession;

namespace PCM {

class ManagerProxy : public ManagerInterface {
    WTF_MAKE_TZONE_ALLOCATED(ManagerProxy);
public:
    static Ref<ManagerProxy> create(const String& machServiceName, NetworkSession&);

    using ApplicationBundleIdentifier = String;

    void storeUnattributed(WebCore::PrivateClickMeasurement&&, CompletionHandler<void()>&&) final;
    void handleAttribution(WebCore::PCM::AttributionTriggerData&&, const URL& requestURL, WebCore::RegistrableDomain&& redirectDomain, const URL& firstPartyURL, const ApplicationBundleIdentifier&) final;
    void clear(CompletionHandler<void()>&&) final;
    void clearForRegistrableDomain(WebCore::RegistrableDomain&&, CompletionHandler<void()>&&) final;
    void migratePrivateClickMeasurementFromLegacyStorage(WebCore::PrivateClickMeasurement&&, PrivateClickMeasurementAttributionType) final;
    void setDebugModeIsEnabled(bool) final;

    void toStringForTesting(CompletionHandler<void(String)>&&) const final;
    void setOverrideTimerForTesting(bool) final;
    void setTokenPublicKeyURLForTesting(URL&&) final;
    void setTokenSignatureURLForTesting(URL&&) final;
    void setAttributionReportURLsForTesting(URL&& sourceURL, URL&& destinationURL) final;
    void markAllUnattributedAsExpiredForTesting() final;
    void markAttributedPrivateClickMeasurementsAsExpiredForTesting(CompletionHandler<void()>&&) final;
    void setPCMFraudPreventionValuesForTesting(String&& unlinkableToken, String&& secretToken, String&& signature, String&& keyID) final;
    void startTimerImmediatelyForTesting() final;
    void setPrivateClickMeasurementAppBundleIDForTesting(ApplicationBundleIdentifier&&) final;
    void destroyStoreForTesting(CompletionHandler<void()>&&) final;
    void allowTLSCertificateChainForLocalPCMTesting(const WebCore::CertificateInfo&) final;

private:
    ManagerProxy(const String& machServiceName, NetworkSession&);

    template<MessageType messageType, typename... Args>
    void sendMessage(Args&&...) const;
    template<MessageType messageType, typename... Args, typename... ReplyArgs>
    void sendMessageWithReply(CompletionHandler<void(ReplyArgs...)>&&, Args&&...) const;

    Ref<Connection> protectedConnection() const;

    Ref<Connection> m_connection;
};

} // namespace PCM

} // namespace WebKit
