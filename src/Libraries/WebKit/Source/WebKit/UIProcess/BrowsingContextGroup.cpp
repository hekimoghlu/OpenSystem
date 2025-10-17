/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 20, 2022.
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
#include "BrowsingContextGroup.h"

#include "FrameProcess.h"
#include "PageLoadState.h"
#include "ProvisionalPageProxy.h"
#include "RemotePageProxy.h"
#include "WebFrameProxy.h"
#include "WebPageProxy.h"
#include "WebProcessProxy.h"

namespace WebKit {

using namespace WebCore;

BrowsingContextGroup::BrowsingContextGroup() = default;

BrowsingContextGroup::~BrowsingContextGroup() = default;

Ref<FrameProcess> BrowsingContextGroup::ensureProcessForSite(const Site& site, WebProcessProxy& process, const WebPreferences& preferences)
{
    if (!site.isEmpty() && preferences.siteIsolationEnabled()) {
        if (auto* existingProcess = processForSite(site)) {
            if (existingProcess->process().coreProcessIdentifier() == process.coreProcessIdentifier())
                return *existingProcess;
        }
    }

    return FrameProcess::create(process, *this, site, preferences);
}

Ref<FrameProcess> BrowsingContextGroup::ensureProcessForConnection(IPC::Connection& connection, WebPageProxy& page, const WebPreferences& preferences)
{
    if (preferences.siteIsolationEnabled()) {
        for (auto& process : m_processMap.values()) {
            if (!process)
                continue;
            if (process->process().hasConnection(connection))
                return *process;
        }
    }
    return FrameProcess::create(page.legacyMainFrameProcess(), *this, Site(URL(page.currentURL())), preferences);
}

FrameProcess* BrowsingContextGroup::processForSite(const Site& site)
{
    auto process = m_processMap.get(site);
    if (!process)
        return nullptr;
    if (process->process().state() == WebProcessProxy::State::Terminated)
        return nullptr;
    return process.get();
}

void BrowsingContextGroup::processDidTerminate(WebPageProxy& page, WebProcessProxy& process)
{
    if (&page.siteIsolatedProcess() == &process)
        m_pages.remove(page);
}

void BrowsingContextGroup::addFrameProcess(FrameProcess& process)
{
    auto& site = process.site();
    ASSERT(site.isEmpty() || !m_processMap.get(site) || m_processMap.get(site)->process().state() == WebProcessProxy::State::Terminated || m_processMap.get(site) == &process);
    m_processMap.set(site, process);
    for (auto& page : m_pages) {
        if (site == Site(URL(page.currentURL())))
            return;
        auto& set = m_remotePages.ensure(page, [] {
            return HashSet<Ref<RemotePageProxy>> { };
        }).iterator->value;
        Ref newRemotePage = RemotePageProxy::create(page, process.process(), site);
        newRemotePage->injectPageIntoNewProcess();
#if ASSERT_ENABLED
        for (auto& existingPage : set) {
            ASSERT(existingPage->process().coreProcessIdentifier() != newRemotePage->process().coreProcessIdentifier() || existingPage->site() != newRemotePage->site());
            ASSERT(existingPage->page() == newRemotePage->page());
        }
#endif
        set.add(WTFMove(newRemotePage));
    }
}

void BrowsingContextGroup::removeFrameProcess(FrameProcess& process)
{
    ASSERT(process.site().isEmpty() || m_processMap.get(process.site()).get() == &process || process.process().state() == WebProcessProxy::State::Terminated);
    m_processMap.remove(process.site());

    m_remotePages.removeIf([&] (auto& pair) {
        auto& set = pair.value;
        set.removeIf([&] (auto& remotePage) {
            if (remotePage->process().coreProcessIdentifier() != process.process().coreProcessIdentifier())
                return false;
            remotePage->removePageFromProcess();
            return true;
        });
        return set.isEmpty();
    });
}

void BrowsingContextGroup::addPage(WebPageProxy& page)
{
    ASSERT(!m_pages.contains(page));
    m_pages.add(page);
    auto& set = m_remotePages.ensure(page, [] {
        return HashSet<Ref<RemotePageProxy>> { };
    }).iterator->value;
    m_processMap.removeIf([&] (auto& pair) {
        auto& site = pair.key;
        auto& process = pair.value;
        if (!process) {
            ASSERT_NOT_REACHED_WITH_MESSAGE("FrameProcess should remove itself in the destructor so we should never find a null WeakPtr");
            return true;
        }

        if (process->process().coreProcessIdentifier() == page.legacyMainFrameProcess().coreProcessIdentifier())
            return false;
        Ref newRemotePage = RemotePageProxy::create(page, process->process(), site);
        newRemotePage->injectPageIntoNewProcess();
#if ASSERT_ENABLED
        for (auto& existingPage : set) {
            ASSERT(existingPage->process().coreProcessIdentifier() != newRemotePage->process().coreProcessIdentifier() || existingPage->site() != newRemotePage->site());
            ASSERT(existingPage->page() == newRemotePage->page());
        }
#endif
        set.add(WTFMove(newRemotePage));
        return false;
    });
}

void BrowsingContextGroup::removePage(WebPageProxy& page)
{
    m_pages.remove(page);

    for (auto& remotePage : m_remotePages.take(page))
        remotePage->removePageFromProcess();
}

void BrowsingContextGroup::forEachRemotePage(const WebPageProxy& page, Function<void(RemotePageProxy&)>&& function)
{
    auto it = m_remotePages.find(page);
    if (it == m_remotePages.end())
        return;
    for (Ref remotePage : it->value)
        function(remotePage);
}

RemotePageProxy* BrowsingContextGroup::remotePageInProcess(const WebPageProxy& page, const WebProcessProxy& process)
{
    auto it = m_remotePages.find(page);
    if (it == m_remotePages.end())
        return nullptr;
    for (Ref remotePage : it->value) {
        if (remotePage->process().coreProcessIdentifier() == process.coreProcessIdentifier())
            return remotePage.ptr();
    }
    return nullptr;
}

RefPtr<RemotePageProxy> BrowsingContextGroup::takeRemotePageInProcessForProvisionalPage(const WebPageProxy& page, const WebProcessProxy& process)
{
    auto it = m_remotePages.find(page);
    if (it == m_remotePages.end())
        return nullptr;
    auto* remotePage = remotePageInProcess(page, process);
    if (!remotePage)
        return nullptr;
    return it->value.take(remotePage);
}

void BrowsingContextGroup::transitionPageToRemotePage(WebPageProxy& page, const Site& openerSite)
{
    auto& set = m_remotePages.ensure(page, [] {
        return HashSet<Ref<RemotePageProxy>> { };
    }).iterator->value;

    Ref newRemotePage = RemotePageProxy::create(page, page.legacyMainFrameProcess(), openerSite, &page.messageReceiverRegistration());
#if ASSERT_ENABLED
    for (auto& existingPage : set) {
        ASSERT(existingPage->process().coreProcessIdentifier() != newRemotePage->process().coreProcessIdentifier() || existingPage->site() != newRemotePage->site());
        ASSERT(existingPage->page() == newRemotePage->page());
    }
#endif
    set.add(WTFMove(newRemotePage));
}

void BrowsingContextGroup::transitionProvisionalPageToRemotePage(ProvisionalPageProxy& page, const Site& provisionalNavigationFailureSite)
{
    auto& set = m_remotePages.ensure(*page.page(), [] {
        return HashSet<Ref<RemotePageProxy>> { };
    }).iterator->value;

    Ref newRemotePage = RemotePageProxy::create(*page.page(), page.process(), provisionalNavigationFailureSite, &page.messageReceiverRegistration());
#if ASSERT_ENABLED
    for (auto& existingPage : set) {
        ASSERT(existingPage->process().coreProcessIdentifier() != newRemotePage->process().coreProcessIdentifier() || existingPage->site() != newRemotePage->site());
        ASSERT(existingPage->page() == newRemotePage->page());
    }
#endif
    set.add(WTFMove(newRemotePage));
}

bool BrowsingContextGroup::hasRemotePages(const WebPageProxy& page)
{
    auto it = m_remotePages.find(page);
    return it != m_remotePages.end() && !it->value.isEmpty();
}

} // namespace WebKit
