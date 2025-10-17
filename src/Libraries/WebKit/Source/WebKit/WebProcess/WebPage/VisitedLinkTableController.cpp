/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 7, 2024.
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
#include "VisitedLinkTableController.h"

#include "VisitedLinkStoreMessages.h"
#include "VisitedLinkTableControllerMessages.h"
#include "WebPage.h"
#include "WebProcess.h"
#include <WebCore/BackForwardCache.h>
#include <wtf/NeverDestroyed.h>

namespace WebKit {
using namespace WebCore;

static HashMap<VisitedLinkTableIdentifier, WeakPtr<VisitedLinkTableController>>& visitedLinkTableControllers()
{
    static NeverDestroyed<HashMap<VisitedLinkTableIdentifier, WeakPtr<VisitedLinkTableController>>> visitedLinkTableControllers;
    RELEASE_ASSERT(isMainRunLoop());
    return visitedLinkTableControllers;
}

Ref<VisitedLinkTableController> VisitedLinkTableController::getOrCreate(VisitedLinkTableIdentifier identifier)
{
    auto& visitedLinkTableControllerPtr = visitedLinkTableControllers().add(identifier, nullptr).iterator->value;
    if (RefPtr ptr = visitedLinkTableControllerPtr.get())
        return *ptr;

    auto visitedLinkTableController = adoptRef(*new VisitedLinkTableController(identifier));
    visitedLinkTableControllerPtr = visitedLinkTableController.get();

    return visitedLinkTableController;
}

VisitedLinkTableController::VisitedLinkTableController(VisitedLinkTableIdentifier identifier)
    : m_identifier(identifier)
{
    WebProcess::singleton().addMessageReceiver(Messages::VisitedLinkTableController::messageReceiverName(), m_identifier, *this);
}

VisitedLinkTableController::~VisitedLinkTableController()
{
    ASSERT(visitedLinkTableControllers().contains(m_identifier));

    WebProcess::singleton().removeMessageReceiver(Messages::VisitedLinkTableController::messageReceiverName(), m_identifier);

    visitedLinkTableControllers().remove(m_identifier);
}

bool VisitedLinkTableController::isLinkVisited(Page&, SharedStringHash linkHash, const URL&, const AtomString&)
{
    return m_visitedLinkTable.contains(linkHash);
}

void VisitedLinkTableController::addVisitedLink(Page& page, SharedStringHash linkHash)
{
    if (m_visitedLinkTable.contains(linkHash))
        return;

    WebProcess::singleton().parentProcessConnection()->send(Messages::VisitedLinkStore::AddVisitedLinkHashFromPage(WebPage::fromCorePage(page)->webPageProxyIdentifier(), linkHash), m_identifier);
}

void VisitedLinkTableController::setVisitedLinkTable(SharedMemory::Handle&& handle)
{
    auto sharedMemory = SharedMemory::map(WTFMove(handle), SharedMemory::Protection::ReadOnly);
    if (!sharedMemory)
        return;

    m_visitedLinkTable.setSharedMemory(sharedMemory.releaseNonNull());

    invalidateStylesForAllLinks();
}

void VisitedLinkTableController::visitedLinkStateChanged(const Vector<WebCore::SharedStringHash>& linkHashes)
{
    for (auto linkHash : linkHashes)
        invalidateStylesForLink(linkHash);
}

void VisitedLinkTableController::allVisitedLinkStateChanged()
{
    invalidateStylesForAllLinks();
}

void VisitedLinkTableController::removeAllVisitedLinks()
{
    m_visitedLinkTable.setSharedMemory(nullptr);

    invalidateStylesForAllLinks();
}

} // namespace WebKit
