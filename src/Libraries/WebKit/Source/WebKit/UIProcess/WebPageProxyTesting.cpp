/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 24, 2022.
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
#include "WebPageProxyTesting.h"

#include "Connection.h"
#include "MessageSenderInlines.h"
#include "NetworkProcessMessages.h"
#include "NetworkProcessProxy.h"
#include "WebBackForwardList.h"
#include "WebFrameProxy.h"
#include "WebPageMessages.h"
#include "WebPageProxy.h"
#include "WebPageTestingMessages.h"
#include "WebProcessProxy.h"
#include <WebCore/IntPoint.h>
#include <wtf/CallbackAggregator.h>
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)
#include "DisplayCaptureSessionManager.h"
#endif

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebPageProxyTesting);

WebPageProxyTesting::WebPageProxyTesting(WebPageProxy& page)
    : m_page(page)
{
}

bool WebPageProxyTesting::sendMessage(UniqueRef<IPC::Encoder>&& encoder, OptionSet<IPC::SendOption> sendOptions)
{
    return protectedPage()->protectedLegacyMainFrameProcess()->sendMessage(WTFMove(encoder), sendOptions);
}

bool WebPageProxyTesting::sendMessageWithAsyncReply(UniqueRef<IPC::Encoder>&& encoder, AsyncReplyHandler handler, OptionSet<IPC::SendOption> sendOptions)
{
    return protectedPage()->protectedLegacyMainFrameProcess()->sendMessage(WTFMove(encoder), sendOptions, WTFMove(handler));
}

IPC::Connection* WebPageProxyTesting::messageSenderConnection() const
{
    return &protectedPage()->legacyMainFrameProcess().connection();
}

uint64_t WebPageProxyTesting::messageSenderDestinationID() const
{
    return protectedPage()->webPageIDInMainFrameProcess().toUInt64();
}

void WebPageProxyTesting::dispatchActivityStateUpdate()
{
    RunLoop::protectedCurrent()->dispatch([protectedPage = protectedPage()] {
        protectedPage->updateActivityState();
        protectedPage->dispatchActivityStateChange();
    });
}

void WebPageProxyTesting::isLayerTreeFrozen(CompletionHandler<void(bool)>&& completionHandler)
{
    sendWithAsyncReply(Messages::WebPageTesting::IsLayerTreeFrozen(), WTFMove(completionHandler));
}

void WebPageProxyTesting::setCrossSiteLoadWithLinkDecorationForTesting(const URL& fromURL, const URL& toURL, bool wasFiltered, CompletionHandler<void()>&& completionHandler)
{
    protectedPage()->protectedWebsiteDataStore()->protectedNetworkProcess()->setCrossSiteLoadWithLinkDecorationForTesting(protectedPage()->sessionID(), WebCore::RegistrableDomain { fromURL }, WebCore::RegistrableDomain { toURL }, wasFiltered, WTFMove(completionHandler));
}

void WebPageProxyTesting::setPermissionLevel(const String& origin, bool allowed)
{
    protectedPage()->forEachWebContentProcess([&](auto& webProcess, auto pageID) {
        webProcess.send(Messages::WebPageTesting::SetPermissionLevel(origin, allowed), pageID);
    });
}

bool WebPageProxyTesting::isEditingCommandEnabled(const String& commandName)
{
    RefPtr focusedOrMainFrame = m_page->focusedOrMainFrame();
    auto targetFrameID = focusedOrMainFrame ? std::optional(focusedOrMainFrame->frameID()) : std::nullopt;
    auto sendResult = protectedPage()->sendSyncToProcessContainingFrame(targetFrameID, Messages::WebPageTesting::IsEditingCommandEnabled(commandName), Seconds::infinity());
    if (!sendResult.succeeded())
        return false;
    auto [result] = sendResult.takeReply();
    return result;
}

void WebPageProxyTesting::dumpPrivateClickMeasurement(CompletionHandler<void(const String&)>&& completionHandler)
{
    protectedPage()->protectedWebsiteDataStore()->protectedNetworkProcess()->sendWithAsyncReply(Messages::NetworkProcess::DumpPrivateClickMeasurement(m_page->websiteDataStore().sessionID()), WTFMove(completionHandler));
}

void WebPageProxyTesting::clearPrivateClickMeasurement(CompletionHandler<void()>&& completionHandler)
{
    protectedPage()->protectedWebsiteDataStore()->protectedNetworkProcess()->sendWithAsyncReply(Messages::NetworkProcess::ClearPrivateClickMeasurement(m_page->websiteDataStore().sessionID()), WTFMove(completionHandler));
}

void WebPageProxyTesting::setPrivateClickMeasurementOverrideTimer(bool value, CompletionHandler<void()>&& completionHandler)
{
    protectedPage()->protectedWebsiteDataStore()->protectedNetworkProcess()->sendWithAsyncReply(Messages::NetworkProcess::SetPrivateClickMeasurementOverrideTimerForTesting(m_page->websiteDataStore().sessionID(), value), WTFMove(completionHandler));
}

void WebPageProxyTesting::markAttributedPrivateClickMeasurementsAsExpired(CompletionHandler<void()>&& completionHandler)
{
    protectedPage()->protectedWebsiteDataStore()->protectedNetworkProcess()->sendWithAsyncReply(Messages::NetworkProcess::MarkAttributedPrivateClickMeasurementsAsExpiredForTesting(m_page->websiteDataStore().sessionID()), WTFMove(completionHandler));
}

void WebPageProxyTesting::setPrivateClickMeasurementEphemeralMeasurement(bool value, CompletionHandler<void()>&& completionHandler)
{
    protectedPage()->protectedWebsiteDataStore()->protectedNetworkProcess()->sendWithAsyncReply(Messages::NetworkProcess::SetPrivateClickMeasurementEphemeralMeasurementForTesting(m_page->websiteDataStore().sessionID(), value), WTFMove(completionHandler));
}

void WebPageProxyTesting::simulatePrivateClickMeasurementSessionRestart(CompletionHandler<void()>&& completionHandler)
{
    protectedPage()->protectedWebsiteDataStore()->protectedNetworkProcess()->sendWithAsyncReply(Messages::NetworkProcess::SimulatePrivateClickMeasurementSessionRestart(m_page->websiteDataStore().sessionID()), WTFMove(completionHandler));
}

void WebPageProxyTesting::setPrivateClickMeasurementTokenPublicKeyURL(const URL& url, CompletionHandler<void()>&& completionHandler)
{
    protectedPage()->protectedWebsiteDataStore()->protectedNetworkProcess()->sendWithAsyncReply(Messages::NetworkProcess::SetPrivateClickMeasurementTokenPublicKeyURLForTesting(m_page->websiteDataStore().sessionID(), url), WTFMove(completionHandler));
}

void WebPageProxyTesting::setPrivateClickMeasurementTokenSignatureURL(const URL& url, CompletionHandler<void()>&& completionHandler)
{
    protectedPage()->protectedWebsiteDataStore()->protectedNetworkProcess()->sendWithAsyncReply(Messages::NetworkProcess::SetPrivateClickMeasurementTokenSignatureURLForTesting(m_page->websiteDataStore().sessionID(), url), WTFMove(completionHandler));
}

void WebPageProxyTesting::setPrivateClickMeasurementAttributionReportURLs(const URL& sourceURL, const URL& destinationURL, CompletionHandler<void()>&& completionHandler)
{
    protectedPage()->protectedWebsiteDataStore()->protectedNetworkProcess()->sendWithAsyncReply(Messages::NetworkProcess::SetPrivateClickMeasurementAttributionReportURLsForTesting(m_page->websiteDataStore().sessionID(), sourceURL, destinationURL), WTFMove(completionHandler));
}

void WebPageProxyTesting::markPrivateClickMeasurementsAsExpired(CompletionHandler<void()>&& completionHandler)
{
    protectedPage()->protectedWebsiteDataStore()->protectedNetworkProcess()->sendWithAsyncReply(Messages::NetworkProcess::MarkPrivateClickMeasurementsAsExpiredForTesting(m_page->websiteDataStore().sessionID()), WTFMove(completionHandler));
}

void WebPageProxyTesting::setPCMFraudPreventionValues(const String& unlinkableToken, const String& secretToken, const String& signature, const String& keyID, CompletionHandler<void()>&& completionHandler)
{
    protectedPage()->protectedWebsiteDataStore()->protectedNetworkProcess()->sendWithAsyncReply(Messages::NetworkProcess::SetPCMFraudPreventionValuesForTesting(m_page->websiteDataStore().sessionID(), unlinkableToken, secretToken, signature, keyID), WTFMove(completionHandler));
}

void WebPageProxyTesting::setPrivateClickMeasurementAppBundleID(const String& appBundleIDForTesting, CompletionHandler<void()>&& completionHandler)
{
    protectedPage()->protectedWebsiteDataStore()->protectedNetworkProcess()->sendWithAsyncReply(Messages::NetworkProcess::SetPrivateClickMeasurementAppBundleIDForTesting(m_page->websiteDataStore().sessionID(), appBundleIDForTesting), WTFMove(completionHandler));
}

#if ENABLE(NOTIFICATIONS)
void WebPageProxyTesting::clearNotificationPermissionState()
{
    send(Messages::WebPageTesting::ClearNotificationPermissionState());
}
#endif

void WebPageProxyTesting::clearWheelEventTestMonitor()
{
    if (!protectedPage()->hasRunningProcess())
        return;
    send(Messages::WebPageTesting::ClearWheelEventTestMonitor());
}

#if PLATFORM(COCOA) && ENABLE(MEDIA_STREAM)
void WebPageProxyTesting::setIndexOfGetDisplayMediaDeviceSelectedForTesting(std::optional<unsigned> index)
{
    DisplayCaptureSessionManager::singleton().setIndexOfDeviceSelectedForTesting(index);
}

void WebPageProxyTesting::setSystemCanPromptForGetDisplayMediaForTesting(bool canPrompt)
{
    DisplayCaptureSessionManager::singleton().setSystemCanPromptForTesting(canPrompt);
}
#endif

void WebPageProxyTesting::setTopContentInset(float contentInset, CompletionHandler<void()>&& completionHandler)
{
    sendWithAsyncReply(Messages::WebPageTesting::SetTopContentInset(contentInset), WTFMove(completionHandler));
}

Ref<WebPageProxy> WebPageProxyTesting::protectedPage() const
{
    return m_page.get();
}

void WebPageProxyTesting::resetStateBetweenTests()
{
    protectedPage()->protectedLegacyMainFrameProcess()->resetState();

    if (RefPtr mainFrame = m_page->mainFrame())
        mainFrame->disownOpener();

    protectedPage()->forEachWebContentProcess([&](auto& webProcess, auto pageID) {
        webProcess.send(Messages::WebPageTesting::ResetStateBetweenTests(), pageID);
    });
}

void WebPageProxyTesting::clearBackForwardList(CompletionHandler<void()>&& completionHandler)
{
    Ref page = m_page.get();
    page->protectedBackForwardList()->clear();

    Ref callbackAggregator = CallbackAggregator::create(WTFMove(completionHandler));
    page->forEachWebContentProcess([&](auto& webProcess, auto pageID) {
        webProcess.sendWithAsyncReply(Messages::WebPageTesting::ClearCachedBackForwardListCounts(), [callbackAggregator] { }, pageID);
    });
}

void WebPageProxyTesting::setTracksRepaints(bool trackRepaints, CompletionHandler<void()>&& completionHandler)
{
    Ref callbackAggregator = CallbackAggregator::create(WTFMove(completionHandler));
    protectedPage()->forEachWebContentProcess([&](auto& webProcess, auto pageID) {
        webProcess.sendWithAsyncReply(Messages::WebPageTesting::SetTracksRepaints(trackRepaints), [callbackAggregator] { }, pageID);
    });
}

void WebPageProxyTesting::displayAndTrackRepaints(CompletionHandler<void()>&& completionHandler)
{
    Ref callbackAggregator = CallbackAggregator::create(WTFMove(completionHandler));
    protectedPage()->forEachWebContentProcess([&](auto& webProcess, auto pageID) {
        webProcess.sendWithAsyncReply(Messages::WebPageTesting::DisplayAndTrackRepaints(), [callbackAggregator] { }, pageID);
    });
}

} // namespace WebKit
