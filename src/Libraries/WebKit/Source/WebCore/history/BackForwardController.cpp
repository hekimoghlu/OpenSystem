/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 3, 2023.
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
#include "BackForwardController.h"

#include "BackForwardClient.h"
#include "HistoryItem.h"
#include "Page.h"
#include "ShouldTreatAsContinuingLoad.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(BackForwardController);

BackForwardController::BackForwardController(Page& page, Ref<BackForwardClient>&& client)
    : m_page(page)
    , m_client(WTFMove(client))
{
}

BackForwardController::~BackForwardController() = default;

RefPtr<HistoryItem> BackForwardController::backItem(std::optional<FrameIdentifier> frameID)
{
    return itemAtIndex(-1, frameID);
}

RefPtr<HistoryItem> BackForwardController::currentItem(std::optional<FrameIdentifier> frameID)
{
    return itemAtIndex(0, frameID);
}

RefPtr<HistoryItem> BackForwardController::forwardItem(std::optional<FrameIdentifier> frameID)
{
    return itemAtIndex(1, frameID);
}

Ref<Page> BackForwardController::protectedPage() const
{
    return m_page.get();
}

Ref<BackForwardClient> BackForwardController::protectedClient() const
{
    return m_client;
}

bool BackForwardController::canGoBackOrForward(int distance) const
{
    if (!distance)
        return true;
    if (distance > 0 && static_cast<unsigned>(distance) <= forwardCount())
        return true;
    if (distance < 0 && static_cast<unsigned>(-distance) <= backCount())
        return true;
    return false;
}

void BackForwardController::goBackOrForward(int distance)
{
    if (!distance)
        return;

    RefPtr historyItem = itemAtIndex(distance);
    if (!historyItem) {
        if (distance > 0) {
            if (int forwardCount = this->forwardCount())
                historyItem = itemAtIndex(forwardCount);
        } else {
            if (int backCount = this->backCount())
                historyItem = itemAtIndex(-backCount);
        }
    }

    if (!historyItem)
        return;

    Ref page { protectedPage() };
    RefPtr localMainFrame = page->localMainFrame();
    if (!localMainFrame)
        return;

    page->goToItem(*localMainFrame, *historyItem, FrameLoadType::IndexedBackForward, ShouldTreatAsContinuingLoad::No);
}

bool BackForwardController::goBack()
{
    RefPtr historyItem = backItem();
    if (!historyItem)
        return false;

    Ref page { protectedPage() };
    RefPtr localMainFrame = page->localMainFrame();
    if (!localMainFrame)
        return false;

    page->goToItem(*localMainFrame, *historyItem, FrameLoadType::Back, ShouldTreatAsContinuingLoad::No);
    return true;
}

bool BackForwardController::goForward()
{
    RefPtr historyItem = forwardItem();
    if (!historyItem)
        return false;

    Ref page { protectedPage() };
    RefPtr localMainFrame = page->localMainFrame();
    if (!localMainFrame)
        return false;

    page->goToItem(*localMainFrame, *historyItem, FrameLoadType::Forward, ShouldTreatAsContinuingLoad::No);
    return true;
}

void BackForwardController::addItem(Ref<HistoryItem>&& item)
{
    protectedClient()->addItem(WTFMove(item));
}

void BackForwardController::setChildItem(BackForwardFrameItemIdentifier frameItemID, Ref<HistoryItem>&& item)
{
    protectedClient()->setChildItem(frameItemID, WTFMove(item));
}

void BackForwardController::setCurrentItem(HistoryItem& item)
{
    protectedClient()->goToItem(item);
}

bool BackForwardController::containsItem(const HistoryItem& item) const
{
    return protectedClient()->containsItem(item);
}

unsigned BackForwardController::count() const
{
    Ref client = m_client;
    return client->backListCount() + 1 + client->forwardListCount();
}

unsigned BackForwardController::backCount() const
{
    return protectedClient()->backListCount();
}

unsigned BackForwardController::forwardCount() const
{
    return protectedClient()->forwardListCount();
}

RefPtr<HistoryItem> BackForwardController::itemAtIndex(int i, std::optional<FrameIdentifier> frameID)
{
    return protectedClient()->itemAtIndex(i, frameID.value_or(m_page->mainFrame().frameID()));
}

Vector<Ref<HistoryItem>> BackForwardController::allItems()
{
    Vector<Ref<HistoryItem>> historyItems;
    for (int index = -1 * static_cast<int>(backCount()); index <= static_cast<int>(forwardCount()); index++) {
        if (RefPtr item = itemAtIndex(index))
            historyItems.append(item.releaseNonNull());
    }

    return historyItems;
}

void BackForwardController::close()
{
    protectedClient()->close();
}

} // namespace WebCore
