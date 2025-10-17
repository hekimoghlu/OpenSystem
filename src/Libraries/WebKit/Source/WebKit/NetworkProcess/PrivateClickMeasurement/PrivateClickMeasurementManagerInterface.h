/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 26, 2025.
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

#include <WebCore/PrivateClickMeasurement.h>
#include <WebCore/RegistrableDomain.h>

namespace WebCore {
class CertificateInfo;
}

namespace WebKit {

namespace Daemon {
class Connection;
}

enum class PrivateClickMeasurementAttributionType : bool { Unattributed, Attributed };

class PrivateClickMeasurementManager;

namespace PCM {

class ManagerInterface : public RefCounted<ManagerInterface> {
public:
    virtual ~ManagerInterface() { };

    using AttributionDestinationSite = WebCore::PCM::AttributionDestinationSite;
    using AttributionTriggerData = WebCore::PCM::AttributionTriggerData;
    using PrivateClickMeasurement = WebCore::PrivateClickMeasurement;
    using RegistrableDomain = WebCore::RegistrableDomain;
    using SourceSite = WebCore::PCM::SourceSite;
    using ApplicationBundleIdentifier = String;

    virtual void storeUnattributed(PrivateClickMeasurement&&, CompletionHandler<void()>&&) = 0;
    virtual void handleAttribution(AttributionTriggerData&&, const URL& requestURL, WebCore::RegistrableDomain&& redirectDomain, const URL& firstPartyURL, const ApplicationBundleIdentifier&) = 0;
    virtual void clear(CompletionHandler<void()>&&) = 0;
    virtual void clearForRegistrableDomain(RegistrableDomain&&, CompletionHandler<void()>&&) = 0;
    virtual void migratePrivateClickMeasurementFromLegacyStorage(PrivateClickMeasurement&&, PrivateClickMeasurementAttributionType) = 0;
    virtual void setDebugModeIsEnabled(bool) = 0;

    virtual void toStringForTesting(CompletionHandler<void(String)>&&) const = 0;
    virtual void setOverrideTimerForTesting(bool value) = 0;
    virtual void setTokenPublicKeyURLForTesting(URL&&) = 0;
    virtual void setTokenSignatureURLForTesting(URL&&) = 0;
    virtual void setAttributionReportURLsForTesting(URL&& sourceURL, URL&& destinationURL) = 0;
    virtual void markAllUnattributedAsExpiredForTesting() = 0;
    virtual void markAttributedPrivateClickMeasurementsAsExpiredForTesting(CompletionHandler<void()>&&) = 0;
    virtual void setPCMFraudPreventionValuesForTesting(String&& unlinkableToken, String&& secretToken, String&& signature, String&& keyID) = 0;
    virtual void startTimerImmediatelyForTesting() = 0;
    virtual void setPrivateClickMeasurementAppBundleIDForTesting(ApplicationBundleIdentifier&&) = 0;
    virtual void destroyStoreForTesting(CompletionHandler<void()>&&) = 0;
    virtual void allowTLSCertificateChainForLocalPCMTesting(const WebCore::CertificateInfo&) = 0;
};

constexpr auto protocolVersionKey { "version"_s };
constexpr uint64_t protocolVersionValue { 1 };

constexpr auto protocolDebugMessageLevelKey { "debug message level"_s };
constexpr auto protocolDebugMessageKey { "debug message"_s };

constexpr auto protocolMessageTypeKey { "message type"_s };
enum class MessageType : uint8_t {
    StoreUnattributed,
    HandleAttribution,
    Clear,
    ClearForRegistrableDomain,
    MigratePrivateClickMeasurementFromLegacyStorage,
    SetDebugModeIsEnabled,
    ToStringForTesting,
    SetOverrideTimerForTesting,
    SetTokenPublicKeyURLForTesting,
    SetTokenSignatureURLForTesting,
    SetAttributionReportURLsForTesting,
    MarkAllUnattributedAsExpiredForTesting,
    MarkAttributedPrivateClickMeasurementsAsExpiredForTesting,
    SetPCMFraudPreventionValuesForTesting,
    StartTimerImmediatelyForTesting,
    SetPrivateClickMeasurementAppBundleIDForTesting,
    DestroyStoreForTesting,
    AllowTLSCertificateChainForLocalPCMTesting
};

constexpr auto protocolEncodedMessageKey { "encoded message"_s };
using EncodedMessage = Vector<uint8_t>;

void decodeMessageAndSendToManager(const Daemon::Connection&, MessageType, std::span<const uint8_t> encodedMessage, CompletionHandler<void(Vector<uint8_t>&&)>&&);
void doDailyActivityInManager();
bool messageTypeSendsReply(MessageType);

void initializePCMStorageInDirectory(const String&);

} // namespace PCM

} // namespace WebKit
