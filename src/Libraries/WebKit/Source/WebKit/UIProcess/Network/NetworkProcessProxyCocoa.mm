/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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
#import "config.h"
#import "NetworkProcessProxy.h"

#import "LaunchServicesDatabaseXPCConstants.h"
#import "NetworkProcessMessages.h"
#import "PageClient.h"
#import "WKUIDelegatePrivate.h"
#import "WKWebViewInternal.h"
#import "WebPageProxy.h"
#import "WebProcessPool.h"
#import "WebProcessProxy.h"
#import "XPCEndpoint.h"
#import <wtf/EnumTraits.h>
#import <wtf/RuntimeApplicationChecks.h>

#if PLATFORM(IOS_FAMILY)
#import <UIKit/UIKit.h>
#import <wtf/BlockPtr.h>
#import <wtf/WeakPtr.h>
#endif

namespace WebKit {

using namespace WebCore;

RefPtr<XPCEventHandler> NetworkProcessProxy::xpcEventHandler() const
{
    return adoptRef(new NetworkProcessProxy::XPCEventHandler(*this));
}

bool NetworkProcessProxy::XPCEventHandler::handleXPCEvent(xpc_object_t event) const
{
    RefPtr networkProcess = m_networkProcess.get();
    if (!networkProcess)
        return false;

    if (!event || xpc_get_type(event) == XPC_TYPE_ERROR)
        return false;

    auto messageName = xpc_dictionary_get_wtfstring(event, XPCEndpoint::xpcMessageNameKey);
    if (messageName.isEmpty())
        return false;

    if (messageName == LaunchServicesDatabaseXPCConstants::xpcLaunchServicesDatabaseXPCEndpointMessageName) {
        networkProcess->m_endpointMessage = event;
        for (auto& processPool : WebProcessPool::allProcessPools()) {
            for (Ref process : processPool->processes())
                networkProcess->sendXPCEndpointToProcess(process);
        }
#if ENABLE(GPU_PROCESS)
        if (RefPtr gpuProcess = GPUProcessProxy::singletonIfCreated())
            networkProcess->sendXPCEndpointToProcess(*gpuProcess);
#endif
    }

    return true;
}

NetworkProcessProxy::XPCEventHandler::XPCEventHandler(const NetworkProcessProxy& networkProcess)
    : m_networkProcess(networkProcess)
{
}

bool NetworkProcessProxy::sendXPCEndpointToProcess(AuxiliaryProcessProxy& process)
{
    RELEASE_LOG(Process, "%p - NetworkProcessProxy::sendXPCEndpointToProcess(%p) state = %d has connection = %d XPC endpoint message = %p", this, &process, enumToUnderlyingType(process.state()), process.hasConnection(), xpcEndpointMessage());

    if (process.state() != AuxiliaryProcessProxy::State::Running)
        return false;
    if (!process.hasConnection())
        return false;
    auto message = xpcEndpointMessage();
    if (!message)
        return false;
    auto xpcConnection = process.connection().xpcConnection();
    RELEASE_ASSERT(xpcConnection);
    xpc_connection_send_message(xpcConnection, message);
    return true;
}

#if PLATFORM(IOS_FAMILY)

void NetworkProcessProxy::addBackgroundStateObservers()
{
    m_backgroundObserver = [[NSNotificationCenter defaultCenter] addObserverForName:UIApplicationDidEnterBackgroundNotification object:[UIApplication sharedApplication] queue:nil usingBlock:makeBlockPtr([weakThis = WeakPtr { *this }](NSNotification *) {
        if (weakThis)
            weakThis->applicationDidEnterBackground();
    }).get()];
    m_foregroundObserver = [[NSNotificationCenter defaultCenter] addObserverForName:UIApplicationWillEnterForegroundNotification object:[UIApplication sharedApplication] queue:nil usingBlock:makeBlockPtr([weakThis = WeakPtr { *this }](NSNotification *) {
        if (weakThis)
            weakThis->applicationWillEnterForeground();
    }).get()];
}

void NetworkProcessProxy::removeBackgroundStateObservers()
{
    [[NSNotificationCenter defaultCenter] removeObserver:m_backgroundObserver.get()];
    [[NSNotificationCenter defaultCenter] removeObserver:m_foregroundObserver.get()];
}

void NetworkProcessProxy::setBackupExclusionPeriodForTesting(PAL::SessionID sessionID, Seconds period, CompletionHandler<void()>&& completionHandler)
{
    sendWithAsyncReply(Messages::NetworkProcess::SetBackupExclusionPeriodForTesting(sessionID, period), WTFMove(completionHandler));
}

#endif

#if ENABLE(APPLE_PAY_REMOTE_UI_USES_SCENE)
void NetworkProcessProxy::getWindowSceneAndBundleIdentifierForPaymentPresentation(WebPageProxyIdentifier webPageProxyIdentifier, CompletionHandler<void(const String&, const String&)>&& completionHandler)
{
    auto sceneIdentifier = nullString();
    auto bundleIdentifier = applicationBundleIdentifier();
    auto page = WebProcessProxy::webPage(webPageProxyIdentifier);
    if (!page || !page->pageClient()) {
        completionHandler(sceneIdentifier, bundleIdentifier);
        return;
    }

    sceneIdentifier = page->pageClient()->sceneID();
    RetainPtr<WKWebView> webView = page->cocoaView();
    id webViewUIDelegate = [webView UIDelegate];
    if ([webViewUIDelegate respondsToSelector:@selector(_hostSceneIdentifierForWebView:)])
        sceneIdentifier = [webViewUIDelegate _hostSceneIdentifierForWebView:webView.get()];
    if ([webViewUIDelegate respondsToSelector:@selector(_hostSceneBundleIdentifierForWebView:)])
        bundleIdentifier = [webViewUIDelegate _hostSceneBundleIdentifierForWebView:webView.get()];

    completionHandler(sceneIdentifier, bundleIdentifier);
}
#endif

#if ENABLE(APPLE_PAY_REMOTE_UI)
void NetworkProcessProxy::getPaymentCoordinatorEmbeddingUserAgent(WebPageProxyIdentifier webPageProxyIdentifier, CompletionHandler<void(const String&)>&& completionHandler)
{
    RefPtr page = WebProcessProxy::webPage(webPageProxyIdentifier);
    if (!page)
        return completionHandler(WebPageProxy::standardUserAgent());

    completionHandler(page->userAgent());
}
#endif

}
