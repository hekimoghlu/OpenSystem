/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 28, 2022.
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
#include "PageConfiguration.h"

#include "AlternativeTextClient.h"
#include "ApplicationCacheStorage.h"
#include "AttachmentElementClient.h"
#include "BackForwardClient.h"
#include "BadgeClient.h"
#include "BroadcastChannelRegistry.h"
#include "CacheStorageProvider.h"
#include "ChromeClient.h"
#include "ContextMenuClient.h"
#include "CookieJar.h"
#include "CryptoClient.h"
#include "DatabaseProvider.h"
#include "DiagnosticLoggingClient.h"
#include "DragClient.h"
#include "EditorClient.h"
#include "Frame.h"
#include "HistoryItem.h"
#include "InspectorClient.h"
#include "LocalFrameLoaderClient.h"
#include "ModelPlayerProvider.h"
#include "PerformanceLoggingClient.h"
#include "PluginInfoProvider.h"
#include "ProcessSyncClient.h"
#include "ProgressTrackerClient.h"
#include "RemoteFrameClient.h"
#include "ScreenOrientationManager.h"
#include "SocketProvider.h"
#include "SpeechRecognitionProvider.h"
#include "SpeechSynthesisClient.h"
#include "StorageNamespaceProvider.h"
#include "StorageProvider.h"
#include "UserContentController.h"
#include "UserContentURLPattern.h"
#include "ValidationMessageClient.h"
#include "VisitedLinkStore.h"
#include "WebRTCProvider.h"
#include <wtf/TZoneMallocInlines.h>
#if ENABLE(WEB_AUTHN)
#include "AuthenticatorCoordinatorClient.h"
#include "CredentialRequestCoordinatorClient.h"
#endif
#if ENABLE(APPLE_PAY)
#include "PaymentCoordinatorClient.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(PageConfiguration);

PageConfiguration::PageConfiguration(
    std::optional<PageIdentifier> identifier,
    PAL::SessionID sessionID,
    UniqueRef<EditorClient>&& editorClient,
    Ref<SocketProvider>&& socketProvider,
    UniqueRef<WebRTCProvider>&& webRTCProvider,
    Ref<CacheStorageProvider>&& cacheStorageProvider,
    Ref<UserContentProvider>&& userContentProvider,
    Ref<BackForwardClient>&& backForwardClient,
    Ref<CookieJar>&& cookieJar,
    UniqueRef<ProgressTrackerClient>&& progressTrackerClient,
    MainFrameCreationParameters&& mainFrameCreationParameters,
    FrameIdentifier mainFrameIdentifier,
    RefPtr<Frame>&& mainFrameOpener,
    UniqueRef<SpeechRecognitionProvider>&& speechRecognitionProvider,
    Ref<BroadcastChannelRegistry>&& broadcastChannelRegistry,
    UniqueRef<StorageProvider>&& storageProvider,
    UniqueRef<ModelPlayerProvider>&& modelPlayerProvider,
    Ref<BadgeClient>&& badgeClient,
    Ref<HistoryItemClient>&& historyItemClient,
#if ENABLE(CONTEXT_MENUS)
    UniqueRef<ContextMenuClient>&& contextMenuClient,
#endif
#if ENABLE(APPLE_PAY)
    Ref<PaymentCoordinatorClient>&& paymentCoordinatorClient,
#endif
    UniqueRef<ChromeClient>&& chromeClient,
    UniqueRef<CryptoClient>&& cryptoClient,
    UniqueRef<ProcessSyncClient>&& processSyncClient
)
    : identifier(identifier)
    , sessionID(sessionID)
    , chromeClient(WTFMove(chromeClient))
#if ENABLE(CONTEXT_MENUS)
    , contextMenuClient(WTFMove(contextMenuClient))
#endif
    , editorClient(WTFMove(editorClient))
    , socketProvider(WTFMove(socketProvider))
#if ENABLE(APPLE_PAY)
    , paymentCoordinatorClient(WTFMove(paymentCoordinatorClient))
#endif
    , webRTCProvider(WTFMove(webRTCProvider))
    , progressTrackerClient(WTFMove(progressTrackerClient))
    , backForwardClient(WTFMove(backForwardClient))
    , cookieJar(WTFMove(cookieJar))
    , mainFrameCreationParameters(WTFMove(mainFrameCreationParameters))
    , mainFrameIdentifier(WTFMove(mainFrameIdentifier))
    , mainFrameOpener(WTFMove(mainFrameOpener))
    , cacheStorageProvider(WTFMove(cacheStorageProvider))
    , userContentProvider(WTFMove(userContentProvider))
    , broadcastChannelRegistry(WTFMove(broadcastChannelRegistry))
    , speechRecognitionProvider(WTFMove(speechRecognitionProvider))
    , storageProvider(WTFMove(storageProvider))
    , modelPlayerProvider(WTFMove(modelPlayerProvider))
    , badgeClient(WTFMove(badgeClient))
    , historyItemClient(WTFMove(historyItemClient))
    , cryptoClient(WTFMove(cryptoClient))
    , processSyncClient(WTFMove(processSyncClient))
{
}

PageConfiguration::~PageConfiguration() = default;
PageConfiguration::PageConfiguration(PageConfiguration&&) = default;

}
