/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 4, 2023.
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
#include "config.h"
#include "PrivateClickMeasurementEphemeralStore.h"

#include "PrivateClickMeasurementDebugInfo.h"
#include "PrivateClickMeasurementManagerInterface.h"
#include <WebCore/PrivateClickMeasurement.h>

namespace WebKit::PCM {

EphemeralStore::EphemeralStore() = default;
EphemeralStore::~EphemeralStore() = default;

void EphemeralStore::insertPrivateClickMeasurement(WebCore::PrivateClickMeasurement&& attribution, PrivateClickMeasurementAttributionType type, CompletionHandler<void()>&& completionHandler)
{
    ASSERT(attribution.isEphemeral() == WebCore::PCM::AttributionEphemeral::Yes);
    ASSERT_UNUSED(type, type == PrivateClickMeasurementAttributionType::Unattributed);
    m_clickMeasurement = WTFMove(attribution);
    completionHandler();
}

void EphemeralStore::markAllUnattributedPrivateClickMeasurementAsExpiredForTesting()
{
    if (m_clickMeasurement && !m_clickMeasurement->attributionTriggerData())
        reset();
}

void EphemeralStore::attributePrivateClickMeasurement(WebCore::PCM::SourceSite&& sourceSite, WebCore::PCM::AttributionDestinationSite&& destinationSite, const ApplicationBundleIdentifier& applicationBundleIdentifier, WebCore::PCM::AttributionTriggerData&& attributionTriggerData, WebCore::PrivateClickMeasurement::IsRunningLayoutTest isLayoutTest, CompletionHandler<void(std::optional<WebCore::PCM::AttributionSecondsUntilSendData>&&, DebugInfo&&)>&& completionHandler)
{
    DebugInfo debugInfo;
    if (!m_clickMeasurement)
        return completionHandler(std::nullopt, WTFMove(debugInfo));

    if (m_clickMeasurement->sourceSite() != sourceSite || m_clickMeasurement->destinationSite() != destinationSite)
        return completionHandler(std::nullopt, WTFMove(debugInfo));

    if (!applicationBundleIdentifier.isEmpty() && m_clickMeasurement->sourceApplicationBundleID() != applicationBundleIdentifier)
        return completionHandler(std::nullopt, WTFMove(debugInfo));

    completionHandler(m_clickMeasurement->attributeAndGetEarliestTimeToSend(WTFMove(attributionTriggerData), isLayoutTest), WTFMove(debugInfo));
}

void EphemeralStore::privateClickMeasurementToStringForTesting(CompletionHandler<void(String)>&& completionHandler) const
{
    if (!m_clickMeasurement)
        return completionHandler("\nNo ephemeral Private Click Measurement data.\n"_s);

    StringBuilder builder;
    builder.append("\nEphemeral Private Click Measurement:\n"_s);
    builder.append("SourceSite: "_s, m_clickMeasurement->sourceSite().registrableDomain.string(), "\n"_s);
    builder.append("DestinationSite: "_s, m_clickMeasurement->destinationSite().registrableDomain.string(), "\n"_s);
    builder.append("SourceID: "_s, m_clickMeasurement->sourceID(), "\n"_s);
    if (auto trigger = m_clickMeasurement->attributionTriggerData()) {
        builder.append("Trigger data: "_s, trigger->data, "\n"_s);
        builder.append("Trigger priority: "_s, trigger->priority, "\n"_s);
    }
    return completionHandler(builder.toString());
}

void EphemeralStore::allAttributedPrivateClickMeasurement(CompletionHandler<void(Vector<WebCore::PrivateClickMeasurement>&&)>&& completionHandler)
{
    if (m_clickMeasurement && m_clickMeasurement->attributionTriggerData())
        completionHandler({ *m_clickMeasurement });
    else
        completionHandler({ });
}

void EphemeralStore::markAttributedPrivateClickMeasurementsAsExpiredForTesting(CompletionHandler<void()>&& completionHandler)
{
    if (m_clickMeasurement && m_clickMeasurement->attributionTriggerData())
        reset();
    completionHandler();
}

void EphemeralStore::clearExpiredPrivateClickMeasurement()
{
    if (!m_clickMeasurement)
        return;

    if (WallTime::now() - m_clickMeasurement->timeOfAdClick() <= WebCore::PrivateClickMeasurement::maxAge())
        return;

    reset();
}

void EphemeralStore::clearPrivateClickMeasurement(CompletionHandler<void()>&& completionHandler)
{
    reset();
    completionHandler();
}

void EphemeralStore::reset()
{
    m_clickMeasurement = std::nullopt;
}

void EphemeralStore::clearPrivateClickMeasurementForRegistrableDomain(WebCore::RegistrableDomain&& domain, CompletionHandler<void()>&& completionHandler)
{
    if (!m_clickMeasurement)
        return completionHandler();

    if (m_clickMeasurement->sourceSite().registrableDomain == domain || m_clickMeasurement->destinationSite().registrableDomain == domain)
        m_clickMeasurement = std::nullopt;

    completionHandler();
}

void EphemeralStore::clearSentAttribution(WebCore::PrivateClickMeasurement&& attributionToClear, WebCore::PCM::AttributionReportEndpoint endpoint)
{
    auto timesToSend = attributionToClear.timesToSend();
    switch (endpoint) {
    case WebCore::PCM::AttributionReportEndpoint::Source:
        timesToSend.sourceEarliestTimeToSend = std::nullopt;
        break;
    case WebCore::PCM::AttributionReportEndpoint::Destination:
        timesToSend.destinationEarliestTimeToSend = std::nullopt;
        break;
    }

    if (!timesToSend.attributionReportEndpoint()) {
        m_clickMeasurement = std::nullopt;
        return;
    }

    attributionToClear.setTimesToSend(WTFMove(timesToSend));
    m_clickMeasurement = WTFMove(attributionToClear);
}

void EphemeralStore::close(CompletionHandler<void()>&& completionHandler)
{
    reset();
    completionHandler();
}

} // namespace WebKit::PCM
