/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
#include "DownloadProxyMap.h"

#include "APIDownloadClient.h"
#include "AuxiliaryProcessProxy.h"
#include "DownloadProxy.h"
#include "DownloadProxyMessages.h"
#include "Logging.h"
#include "MessageReceiverMap.h"
#include "NetworkProcessMessages.h"
#include "NetworkProcessProxy.h"
#include "ProcessAssertion.h"
#include "WebProcessPool.h"
#include <wtf/StdLibExtras.h>
#include <wtf/TZoneMallocInlines.h>

#if PLATFORM(COCOA)
#include <wtf/cocoa/Entitlements.h>
#endif

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(DownloadProxyMap);

DownloadProxyMap::DownloadProxyMap(NetworkProcessProxy& process)
    : m_process(process)
#if PLATFORM(COCOA) && !HAVE(MODERN_DOWNLOADPROGRESS)
    , m_shouldTakeAssertion(WTF::processHasEntitlement("com.apple.multitasking.systemappassertions"_s))
#endif
{
    platformCreate();
}

DownloadProxyMap::~DownloadProxyMap()
{
    ASSERT(m_downloads.isEmpty());
    platformDestroy();
}

void DownloadProxyMap::ref() const
{
    m_process->ref();
}

void DownloadProxyMap::deref() const
{
    m_process->deref();
}

Ref<NetworkProcessProxy> DownloadProxyMap::protectedProcess()
{
    return m_process.get();
}

#if !PLATFORM(COCOA)
void DownloadProxyMap::platformCreate()
{
}

void DownloadProxyMap::platformDestroy()
{
}
#endif

Ref<DownloadProxy> DownloadProxyMap::createDownloadProxy(WebsiteDataStore& dataStore, Ref<API::DownloadClient>&& client, const WebCore::ResourceRequest& resourceRequest, const FrameInfoData& frameInfo, WebPageProxy* originatingPage)
{
    auto downloadProxy = DownloadProxy::create(*this, dataStore, WTFMove(client), resourceRequest, frameInfo, originatingPage);
    m_downloads.set(downloadProxy->downloadID(), downloadProxy.copyRef());

    RELEASE_LOG(Loading, "Adding download %" PRIu64 " to UIProcess DownloadProxyMap", downloadProxy->downloadID().toUInt64());

    if (m_downloads.size() == 1 && m_shouldTakeAssertion) {
        ASSERT(!m_downloadUIAssertion);
        m_downloadUIAssertion = ProcessAssertion::create(getCurrentProcessID(), "WebKit downloads"_s, ProcessAssertionType::UnboundedNetworking);
        RELEASE_LOG(ProcessSuspension, "UIProcess took 'WebKit downloads' assertion for UIProcess");
    }

    protectedProcess()->addMessageReceiver(Messages::DownloadProxy::messageReceiverName(), downloadProxy->downloadID().toUInt64(), downloadProxy.get());

    return downloadProxy;
}

void DownloadProxyMap::downloadFinished(DownloadProxy& downloadProxy)
{
    auto downloadID = downloadProxy.downloadID();

    RELEASE_LOG(Loading, "Removing download %" PRIu64 " from UIProcess DownloadProxyMap", downloadID.toUInt64());

    ASSERT(m_downloads.contains(downloadID));

    protectedProcess()->removeMessageReceiver(Messages::DownloadProxy::messageReceiverName(), downloadID.toUInt64());
    downloadProxy.invalidate();
    m_downloads.remove(downloadID);

    if (m_downloads.isEmpty() && m_shouldTakeAssertion) {
        ASSERT(m_downloadUIAssertion);
        m_downloadUIAssertion = nullptr;
        RELEASE_LOG(ProcessSuspension, "UIProcess released 'WebKit downloads' assertion for UIProcess");
    }
}

void DownloadProxyMap::invalidate()
{
    // Invalidate all outstanding downloads.
    for (const auto& download : m_downloads.values()) {
        download->processDidClose();
        download->invalidate();
        protectedProcess()->removeMessageReceiver(Messages::DownloadProxy::messageReceiverName(), download->downloadID().toUInt64());
    }

    m_downloads.clear();
    m_downloadUIAssertion = nullptr;
    RELEASE_LOG(ProcessSuspension, "UIProcess DownloadProxyMap invalidated - Released 'WebKit downloads' assertion for UIProcess");
}

} // namespace WebKit
