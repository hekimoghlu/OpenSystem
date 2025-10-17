/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 12, 2025.
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
#include "VisitedLinkStore.h"

#include "VisitedLinkStoreMessages.h"
#include "VisitedLinkTableControllerMessages.h"
#include "WebPageProxy.h"
#include "WebProcessMessages.h"
#include "WebProcessPool.h"
#include "WebProcessProxy.h"

namespace WebKit {
using namespace WebCore;

Ref<VisitedLinkStore> VisitedLinkStore::create()
{
    return adoptRef(*new VisitedLinkStore);
}

VisitedLinkStore::~VisitedLinkStore()
{
    RELEASE_ASSERT(m_processes.isEmptyIgnoringNullReferences());
}

VisitedLinkStore::VisitedLinkStore()
    : m_linkHashStore(*this)
{
}

void VisitedLinkStore::addProcess(WebProcessProxy& process)
{
    ASSERT(!m_processes.contains(process));

    if (!m_processes.add(process).isNewEntry)
        return;

    process.addMessageReceiver(Messages::VisitedLinkStore::messageReceiverName(), identifier(), *this);

    if (m_linkHashStore.isEmpty())
        return;

    sendStoreHandleToProcess(process);
}

void VisitedLinkStore::removeProcess(WebProcessProxy& process)
{
    ASSERT(m_processes.contains(process));
    if (!m_processes.remove(process))
        return;

    process.removeMessageReceiver(Messages::VisitedLinkStore::messageReceiverName(), identifier());
}

void VisitedLinkStore::addVisitedLinkHash(SharedStringHash linkHash)
{
    m_linkHashStore.scheduleAddition(linkHash);
}

bool VisitedLinkStore::containsVisitedLinkHash(WebCore::SharedStringHash linkHash)
{
    return m_linkHashStore.contains(linkHash);
}

void VisitedLinkStore::removeVisitedLinkHash(WebCore::SharedStringHash linkHash)
{
    m_linkHashStore.scheduleRemoval(linkHash);
}

void VisitedLinkStore::removeAll()
{
    m_linkHashStore.clear();

    for (Ref process : m_processes) {
        ASSERT(process->processPool().processes().containsIf([&](auto& item) { return item.ptr() == &process.get(); }));
        process->send(Messages::VisitedLinkTableController::RemoveAllVisitedLinks(), identifier());
    }
}

void VisitedLinkStore::addVisitedLinkHashFromPage(WebPageProxyIdentifier pageProxyID, SharedStringHash linkHash)
{
    if (RefPtr page = WebProcessProxy::webPage(pageProxyID)) {
        if (!page || !page->addsVisitedLinks())
            return;
    }

    addVisitedLinkHash(linkHash);
}

void VisitedLinkStore::sendStoreHandleToProcess(WebProcessProxy& process)
{
    ASSERT(process.processPool().processes().containsIf([&](auto& item) { return item.ptr() == &process; }));

    auto handle = m_linkHashStore.createSharedMemoryHandle();
    if (!handle)
        return;
    process.send(Messages::VisitedLinkTableController::SetVisitedLinkTable(WTFMove(*handle)), identifier());
}

void VisitedLinkStore::didInvalidateSharedMemory()
{
    for (Ref process : m_processes)
        sendStoreHandleToProcess(process.get());
}

void VisitedLinkStore::didUpdateSharedStringHashes(const Vector<WebCore::SharedStringHash>& addedHashes, const Vector<WebCore::SharedStringHash>& removedHashes)
{
    ASSERT(!addedHashes.isEmpty() || !removedHashes.isEmpty());

    for (Ref process : m_processes) {
        ASSERT(process->processPool().processes().containsIf([&](auto& item) { return item.ptr() == process.ptr(); }));

        if (addedHashes.size() > 20 || !removedHashes.isEmpty() || process->throttler().isSuspended())
            process->send(Messages::VisitedLinkTableController::AllVisitedLinkStateChanged(), identifier());
        else
            process->send(Messages::VisitedLinkTableController::VisitedLinkStateChanged(addedHashes), identifier());
    }
}

} // namespace WebKit
