/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 4, 2023.
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

#include "NetworkLoadParameters.h"
#include "NetworkProcess.h"
#include "PrivateClickMeasurementClient.h"
#include "PrivateClickMeasurementManagerInterface.h"
#include "PrivateClickMeasurementStore.h"
#include <WebCore/Timer.h>
#include <wtf/CompletionHandler.h>
#include <wtf/JSONValues.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebKit {

class PrivateClickMeasurementManager : public PCM::ManagerInterface, public CanMakeWeakPtr<PrivateClickMeasurementManager> {
    WTF_MAKE_TZONE_ALLOCATED(PrivateClickMeasurementManager);
public:
    static Ref<PrivateClickMeasurementManager> create(UniqueRef<PCM::Client>&&, const String& storageDirectory);

    ~PrivateClickMeasurementManager();

    using ApplicationBundleIdentifier = String;

    void storeUnattributed(PrivateClickMeasurement&&, CompletionHandler<void()>&&) final;
    void handleAttribution(AttributionTriggerData&&, const URL& requestURL, WebCore::RegistrableDomain&& redirectDomain, const URL& firstPartyURL, const ApplicationBundleIdentifier&) final;
    void clear(CompletionHandler<void()>&&) final;
    void clearForRegistrableDomain(RegistrableDomain&&, CompletionHandler<void()>&&) final;
    void migratePrivateClickMeasurementFromLegacyStorage(PrivateClickMeasurement&&, PrivateClickMeasurementAttributionType) final;
    void setDebugModeIsEnabled(bool) final;
    void firePendingAttributionRequests();

    void toStringForTesting(CompletionHandler<void(String)>&&) const final;
    void setOverrideTimerForTesting(bool value) final { m_isRunningTest = value; }
    void setTokenPublicKeyURLForTesting(URL&&) final;
    void setTokenSignatureURLForTesting(URL&&) final;
    void setAttributionReportURLsForTesting(URL&& sourceURL, URL&& destinationURL) final;
    void markAllUnattributedAsExpiredForTesting() final;
    void markAttributedPrivateClickMeasurementsAsExpiredForTesting(CompletionHandler<void()>&&) final;
    void setPCMFraudPreventionValuesForTesting(String&& unlinkableToken, String&& secretToken, String&& signature, String&& keyID) final;
    void startTimerImmediatelyForTesting() final;
    void setPrivateClickMeasurementAppBundleIDForTesting(ApplicationBundleIdentifier&&);
    void destroyStoreForTesting(CompletionHandler<void()>&&) final;
    void allowTLSCertificateChainForLocalPCMTesting(const WebCore::CertificateInfo&) final;

private:
    PrivateClickMeasurementManager(UniqueRef<PCM::Client>&&, const String& storageDirectory);

    PCM::Store& store();
    const PCM::Store& store() const;
    void initializeStore() const;
    void startTimer(Seconds);
    void getTokenPublicKey(PrivateClickMeasurement&&, WebCore::PCM::AttributionReportEndpoint, PrivateClickMeasurement::PcmDataCarried, Function<void(PrivateClickMeasurement&& attribution, const String& publicKeyBase64URL)>&&);
    void getTokenPublicKey(AttributionTriggerData&&, WebCore::PCM::AttributionReportEndpoint, PrivateClickMeasurement::PcmDataCarried, Function<void(AttributionTriggerData&&, const String& publicKeyBase64URL)>&&);
    void configureForTokenSigning(PrivateClickMeasurement::PcmDataCarried&, URL& tokenSignatureURL, std::optional<URL> givenTokenSignatureURL);
    std::optional<String> getSignatureBase64URLFromTokenSignatureResponse(const String& errorDescription, const RefPtr<JSON::Object>&);
    void getSignedUnlinkableTokenForSource(PrivateClickMeasurement&&);
    void getSignedUnlinkableTokenForDestination(SourceSite&&, AttributionDestinationSite&&, AttributionTriggerData&&, const ApplicationBundleIdentifier&);
    void insertPrivateClickMeasurement(PrivateClickMeasurement&&, PrivateClickMeasurementAttributionType, CompletionHandler<void()>&&);
    void clearSentAttribution(PrivateClickMeasurement&&, WebCore::PCM::AttributionReportEndpoint);
    void attribute(SourceSite&&, AttributionDestinationSite&&, AttributionTriggerData&&, const ApplicationBundleIdentifier&);
    void fireConversionRequest(const PrivateClickMeasurement&, WebCore::PCM::AttributionReportEndpoint);
    void fireConversionRequestImpl(const PrivateClickMeasurement&, WebCore::PCM::AttributionReportEndpoint);
    void clearExpired();
    bool featureEnabled() const;
    bool debugModeEnabled() const;
    Seconds randomlyBetweenFifteenAndThirtyMinutes() const;

    RunLoop::Timer m_firePendingAttributionRequestsTimer;
    bool m_isRunningTest { false };
    std::optional<URL> m_tokenPublicKeyURLForTesting;
    std::optional<URL> m_tokenSignatureURLForTesting;
    std::optional<ApplicationBundleIdentifier> m_privateClickMeasurementAppBundleIDForTesting;
    mutable RefPtr<PCM::Store> m_store;
    String m_storageDirectory;
    UniqueRef<PCM::Client> m_client;

    struct AttributionReportTestConfig {
        URL attributionReportClickSourceURL;
        URL attributionReportClickDestinationURL;
    };

    std::optional<AttributionReportTestConfig> m_attributionReportTestConfig;

    struct TestingFraudPreventionValues {
        String unlinkableTokenForSource;
        String secretTokenForSource;
        String signatureForSource;
        String keyIDForSource;
        String unlinkableTokenForDestination;
        String secretTokenForDestination;
        String signatureForDestination;
        String keyIDForDestination;
    };

    std::optional<TestingFraudPreventionValues> m_fraudPreventionValuesForTesting;
};
    
} // namespace WebKit
