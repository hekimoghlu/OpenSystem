/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 8, 2023.
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
#include "PrivateClickMeasurementManagerProxy.h"

#include "DaemonDecoder.h"
#include "DaemonEncoder.h"
#include "PrivateClickMeasurementConnection.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebKit::PCM {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ManagerProxy);

template<MessageType messageType, typename... Args>
void ManagerProxy::sendMessage(Args&&... args) const
{
    Daemon::Encoder encoder;
    encoder.encode(std::forward<Args>(args)...);
    protectedConnection()->send(messageType, encoder.takeBuffer());
}

template<typename... Args> struct ReplyCaller;
template<> struct ReplyCaller<> {
    static void callReply(Daemon::Decoder&& decoder, CompletionHandler<void()>&& completionHandler)
    {
        completionHandler();
    }
};
template<> struct ReplyCaller<String> {
    static void callReply(Daemon::Decoder&& decoder, CompletionHandler<void(String&&)>&& completionHandler)
    {
        std::optional<String> string;
        decoder >> string;
        if (!string)
            return completionHandler({ });
        completionHandler(WTFMove(*string));
    }
};

template<MessageType messageType, typename... Args, typename... ReplyArgs>
void ManagerProxy::sendMessageWithReply(CompletionHandler<void(ReplyArgs...)>&& completionHandler, Args&&... args) const
{
    Daemon::Encoder encoder;
    encoder.encode(std::forward<Args>(args)...);
    protectedConnection()->sendWithReply(messageType, encoder.takeBuffer(), [completionHandler = WTFMove(completionHandler)] (auto replyBuffer) mutable {
        Daemon::Decoder decoder(WTFMove(replyBuffer));
        ReplyCaller<ReplyArgs...>::callReply(WTFMove(decoder), WTFMove(completionHandler));
    });
}

Ref<ManagerProxy> ManagerProxy::create(const String& machServiceName, NetworkSession& networkSession)
{
    return adoptRef(*new ManagerProxy(machServiceName, networkSession));
}

ManagerProxy::ManagerProxy(const String& machServiceName, NetworkSession& networkSession)
    : m_connection(Connection::create(machServiceName.utf8(), networkSession))
{ }

Ref<Connection> ManagerProxy::protectedConnection() const
{
    return m_connection;
}

void ManagerProxy::storeUnattributed(WebCore::PrivateClickMeasurement&& pcm, CompletionHandler<void()>&& completionHandler)
{
    sendMessageWithReply<MessageType::StoreUnattributed>(WTFMove(completionHandler), pcm);
}

void ManagerProxy::handleAttribution(WebCore::PCM::AttributionTriggerData&& triggerData, const URL& requestURL, WebCore::RegistrableDomain&& redirectDomain, const URL& firstPartyURL, const ApplicationBundleIdentifier& applicationBundleIdentifier)
{
    sendMessage<MessageType::HandleAttribution>(triggerData, requestURL, redirectDomain, firstPartyURL, applicationBundleIdentifier);
}

void ManagerProxy::clear(CompletionHandler<void()>&& completionHandler)
{
    sendMessageWithReply<MessageType::Clear>(WTFMove(completionHandler));
}

void ManagerProxy::clearForRegistrableDomain(WebCore::RegistrableDomain&& domain, CompletionHandler<void()>&& completionHandler)
{
    sendMessageWithReply<MessageType::ClearForRegistrableDomain>(WTFMove(completionHandler), domain);
}

void ManagerProxy::setDebugModeIsEnabled(bool enabled)
{
    sendMessage<MessageType::SetDebugModeIsEnabled>(enabled);
}

void ManagerProxy::migratePrivateClickMeasurementFromLegacyStorage(WebCore::PrivateClickMeasurement&& pcm, PrivateClickMeasurementAttributionType type)
{
    sendMessage<MessageType::MigratePrivateClickMeasurementFromLegacyStorage>(pcm, type);
}

void ManagerProxy::toStringForTesting(CompletionHandler<void(String)>&& completionHandler) const
{
    sendMessageWithReply<MessageType::ToStringForTesting>(WTFMove(completionHandler));
}

void ManagerProxy::setOverrideTimerForTesting(bool value)
{
    sendMessage<MessageType::SetOverrideTimerForTesting>(value);
}

void ManagerProxy::setTokenPublicKeyURLForTesting(URL&& url)
{
    sendMessage<MessageType::SetTokenPublicKeyURLForTesting>(url);
}

void ManagerProxy::setTokenSignatureURLForTesting(URL&& url)
{
    sendMessage<MessageType::SetTokenSignatureURLForTesting>(url);
}

void ManagerProxy::setAttributionReportURLsForTesting(URL&& sourceURL, URL&& destinationURL)
{
    sendMessage<MessageType::SetAttributionReportURLsForTesting>(sourceURL, destinationURL);
}

void ManagerProxy::markAllUnattributedAsExpiredForTesting()
{
    sendMessage<MessageType::MarkAllUnattributedAsExpiredForTesting>();
}

void ManagerProxy::markAttributedPrivateClickMeasurementsAsExpiredForTesting(CompletionHandler<void()>&& completionHandler)
{
    sendMessageWithReply<MessageType::MarkAttributedPrivateClickMeasurementsAsExpiredForTesting>(WTFMove(completionHandler));
}

void ManagerProxy::setPCMFraudPreventionValuesForTesting(String&& unlinkableToken, String&& secretToken, String&& signature, String&& keyID)
{
    sendMessage<MessageType::SetPCMFraudPreventionValuesForTesting>(unlinkableToken, secretToken, signature, keyID);
}

void ManagerProxy::startTimerImmediatelyForTesting()
{
    sendMessage<MessageType::StartTimerImmediatelyForTesting>();
}

void ManagerProxy::setPrivateClickMeasurementAppBundleIDForTesting(ApplicationBundleIdentifier&& appBundleID)
{
    sendMessage<MessageType::SetPrivateClickMeasurementAppBundleIDForTesting>(appBundleID);
}

void ManagerProxy::destroyStoreForTesting(CompletionHandler<void()>&& completionHandler)
{
    sendMessageWithReply<MessageType::DestroyStoreForTesting>(WTFMove(completionHandler));
}

void ManagerProxy::allowTLSCertificateChainForLocalPCMTesting(const WebCore::CertificateInfo& certificateInfo)
{
    sendMessage<MessageType::AllowTLSCertificateChainForLocalPCMTesting>(certificateInfo);
}

} // namespace WebKit::PCM
