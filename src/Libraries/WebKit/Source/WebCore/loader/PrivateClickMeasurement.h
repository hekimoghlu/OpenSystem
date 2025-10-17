/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 29, 2023.
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

#include "AttributionSecondsUntilSendData.h"
#include "AttributionTimeToSendData.h"
#include "AttributionTriggerData.h"
#include "EphemeralNonce.h"
#include "PCMSites.h"
#include "RegistrableDomain.h"
#include <wtf/CompletionHandler.h>
#include <wtf/text/Base64.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

namespace PCM {
enum class AttributionEphemeral : bool { No, Yes };
}

class PrivateClickMeasurement {
public:
    enum class PcmDataCarried : bool { NonPersonallyIdentifiable, PersonallyIdentifiable };
    enum class IsRunningLayoutTest : bool { No, Yes };

    using SourceID = uint8_t;

    PrivateClickMeasurement(SourceID sourceID, const PCM::SourceSite& sourceSite, const PCM::AttributionDestinationSite& destinationSite, const String& sourceApplicationBundleID, WallTime timeOfAdClick, PCM::AttributionEphemeral isEphemeral)
        : m_sourceID { sourceID }
        , m_sourceSite { sourceSite }
        , m_destinationSite { destinationSite }
        , m_timeOfAdClick { timeOfAdClick }
        , m_isEphemeral { isEphemeral }
        , m_sourceApplicationBundleID { sourceApplicationBundleID }
    {
    }

    PrivateClickMeasurement(SourceID&& sourceID, PCM::SourceSite&& sourceSite, PCM::AttributionDestinationSite&& destinationSite, WallTime&& timeOfAdClick, PCM::AttributionEphemeral&& isEphemeral, std::optional<uint64_t>&& adamID, std::optional<PCM::AttributionTriggerData>&& attributionTriggerData, PCM::AttributionTimeToSendData&& timesToSend, std::optional<PCM::EphemeralNonce>&& ephemeralSourceNonce, String&& sourceApplicationBundleID)
        : m_sourceID(WTFMove(sourceID))
        , m_sourceSite(WTFMove(sourceSite))
        , m_destinationSite(WTFMove(destinationSite))
        , m_timeOfAdClick(WTFMove(timeOfAdClick))
        , m_isEphemeral(WTFMove(isEphemeral))
        , m_adamID(WTFMove(adamID))
        , m_attributionTriggerData(WTFMove(attributionTriggerData))
        , m_timesToSend(WTFMove(timesToSend))
        , m_ephemeralSourceNonce(WTFMove(ephemeralSourceNonce))
        , m_sourceApplicationBundleID(WTFMove(sourceApplicationBundleID))
    {
    }
    
    WEBCORE_EXPORT static const Seconds maxAge();
    WEBCORE_EXPORT bool isNeitherSameSiteNorCrossSiteTriggeringEvent(const RegistrableDomain& redirectDomain, const URL& firstPartyURL, const PCM::AttributionTriggerData&);
    WEBCORE_EXPORT static Expected<PCM::AttributionTriggerData, String> parseAttributionRequest(const URL& redirectURL);
    WEBCORE_EXPORT PCM::AttributionSecondsUntilSendData attributeAndGetEarliestTimeToSend(PCM::AttributionTriggerData&&, IsRunningLayoutTest);
    WEBCORE_EXPORT bool hasHigherPriorityThan(const PrivateClickMeasurement&) const;
    WEBCORE_EXPORT URL attributionReportClickSourceURL() const;
    WEBCORE_EXPORT URL attributionReportClickDestinationURL() const;
    WEBCORE_EXPORT Ref<JSON::Object> attributionReportJSON() const;
    const PCM::SourceSite& sourceSite() const { return m_sourceSite; };
    const PCM::AttributionDestinationSite& destinationSite() const { return m_destinationSite; };
    WallTime timeOfAdClick() const { return m_timeOfAdClick; }
    WEBCORE_EXPORT bool hasPreviouslyBeenReported();
    PCM::AttributionTimeToSendData timesToSend() const { return m_timesToSend; };
    void setTimesToSend(PCM::AttributionTimeToSendData data) { m_timesToSend = data; }
    const SourceID& sourceID() const { return m_sourceID; }
    const std::optional<PCM::AttributionTriggerData>& attributionTriggerData() const { return m_attributionTriggerData; }
    void setAttribution(PCM::AttributionTriggerData&& attributionTriggerData) { m_attributionTriggerData = WTFMove(attributionTriggerData); }
    const String& sourceApplicationBundleID() const { return m_sourceApplicationBundleID; }
    WEBCORE_EXPORT void setSourceApplicationBundleIDForTesting(const String&);

    PCM::AttributionEphemeral isEphemeral() const { return m_isEphemeral; }
    void setEphemeral(PCM::AttributionEphemeral isEphemeral) { m_isEphemeral = isEphemeral; }

    // MARK: - Fraud Prevention
    WEBCORE_EXPORT const std::optional<const URL> tokenPublicKeyURL() const;
    WEBCORE_EXPORT static const std::optional<const URL> tokenPublicKeyURL(const RegistrableDomain&);
    WEBCORE_EXPORT const std::optional<const URL> tokenSignatureURL() const;
    WEBCORE_EXPORT static const std::optional<const URL> tokenSignatureURL(const RegistrableDomain&);

    WEBCORE_EXPORT Ref<JSON::Object> tokenSignatureJSON() const;

    WEBCORE_EXPORT void setEphemeralSourceNonce(PCM::EphemeralNonce&&);
    std::optional<PCM::EphemeralNonce> ephemeralSourceNonce() const { return m_ephemeralSourceNonce; };
    void clearEphemeralSourceNonce() { m_ephemeralSourceNonce.reset(); };

#if PLATFORM(COCOA)
    WEBCORE_EXPORT std::optional<String> calculateAndUpdateSourceUnlinkableToken(const String& serverPublicKeyBase64URL);
    WEBCORE_EXPORT static Expected<PCM::DestinationUnlinkableToken, String> calculateAndUpdateDestinationUnlinkableToken(const String& serverPublicKeyBase64URL);
    WEBCORE_EXPORT std::optional<String> calculateAndUpdateSourceSecretToken(const String& serverResponseBase64URL);
    WEBCORE_EXPORT static Expected<PCM::DestinationSecretToken, String> calculateAndUpdateDestinationSecretToken(const String& serverResponseBase64URL, PCM::DestinationUnlinkableToken&);
#endif

    PCM::SourceUnlinkableToken& sourceUnlinkableToken() { return m_sourceUnlinkableToken; }
    void setSourceUnlinkableTokenValue(const String& value) { m_sourceUnlinkableToken.valueBase64URL = value; }
    const std::optional<PCM::SourceSecretToken>& sourceSecretToken() const { return m_sourceSecretToken; }
    WEBCORE_EXPORT void setSourceSecretToken(PCM::SourceSecretToken&&);
    WEBCORE_EXPORT void setDestinationSecretToken(PCM::DestinationSecretToken&&);

    static std::optional<uint64_t> appStoreURLAdamID(const URL&);
    bool isSKAdNetworkAttribution() const { return !!m_adamID; }
    std::optional<uint64_t> adamID() const { return m_adamID; };
    void setAdamID(uint64_t adamID) { m_adamID = adamID; };

    WEBCORE_EXPORT PrivateClickMeasurement isolatedCopy() const &;
    WEBCORE_EXPORT PrivateClickMeasurement isolatedCopy() &&;

private:
    static Expected<PCM::AttributionTriggerData, String> parseAttributionRequestQuery(const URL&);
    bool isValid() const;

#if PLATFORM(COCOA)
    static std::optional<String> calculateAndUpdateUnlinkableToken(const String& serverPublicKeyBase64URL, PCM::UnlinkableToken&, const String& contextForLogMessage);
    static std::optional<String> calculateAndUpdateSecretToken(const String& serverResponseBase64URL, PCM::UnlinkableToken&, PCM::SecretToken&, const String& contextForLogMessage);
#endif

    SourceID m_sourceID;
    PCM::SourceSite m_sourceSite;
    PCM::AttributionDestinationSite m_destinationSite;
    WallTime m_timeOfAdClick;
    PCM::AttributionEphemeral m_isEphemeral;
    std::optional<uint64_t> m_adamID;

    std::optional<PCM::AttributionTriggerData> m_attributionTriggerData;
    PCM::AttributionTimeToSendData m_timesToSend;

    std::optional<PCM::EphemeralNonce> m_ephemeralSourceNonce;
    PCM::SourceUnlinkableToken m_sourceUnlinkableToken;
    std::optional<PCM::SourceSecretToken> m_sourceSecretToken;
    String m_sourceApplicationBundleID;
};

} // namespace WebCore
