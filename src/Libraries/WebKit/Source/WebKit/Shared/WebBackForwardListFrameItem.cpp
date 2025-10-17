/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 3, 2023.
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
#include "WebBackForwardListFrameItem.h"

#include "SessionState.h"
#include "WebBackForwardListItem.h"

namespace WebKit {
using namespace WebCore;

Ref<WebBackForwardListFrameItem> WebBackForwardListFrameItem::create(WebBackForwardListItem& item, WebBackForwardListFrameItem* parentItem, Ref<FrameState>&& frameState)
{
    return adoptRef(*new WebBackForwardListFrameItem(item, parentItem, WTFMove(frameState)));
}

WebBackForwardListFrameItem::WebBackForwardListFrameItem(WebBackForwardListItem& item, WebBackForwardListFrameItem* parentItem, Ref<FrameState>&& frameState)
    : m_backForwardListItem(item)
    , m_identifier(*frameState->frameItemID)
    , m_frameState(WTFMove(frameState))
    , m_parent(parentItem)
{
    m_frameState->itemID = item.identifier();
    auto result = allItems().add({ *m_frameState->frameItemID, *m_frameState->itemID }, *this);
    ASSERT_UNUSED(result, result.isNewEntry);
    for (auto& child : std::exchange(m_frameState->children, { }))
        m_children.append(WebBackForwardListFrameItem::create(item, this, WTFMove(child)));
}

WebBackForwardListFrameItem::~WebBackForwardListFrameItem()
{
    ASSERT(allItems().get({ *m_frameState->frameItemID, *m_frameState->itemID }) == this);
    allItems().remove({ *m_frameState->frameItemID, *m_frameState->itemID });
}

HashMap<std::pair<BackForwardFrameItemIdentifier, BackForwardItemIdentifier>, WeakRef<WebBackForwardListFrameItem>>& WebBackForwardListFrameItem::allItems()
{
    static MainThreadNeverDestroyed<HashMap<std::pair<BackForwardFrameItemIdentifier, BackForwardItemIdentifier>, WeakRef<WebBackForwardListFrameItem>>> items;
    return items;
}

WebBackForwardListFrameItem* WebBackForwardListFrameItem::itemForID(BackForwardItemIdentifier itemID, BackForwardFrameItemIdentifier frameItemID)
{
    return allItems().get({ frameItemID, itemID });
}

std::optional<FrameIdentifier> WebBackForwardListFrameItem::frameID() const
{
    return m_frameState->frameID;
}

const String& WebBackForwardListFrameItem::url() const
{
    return m_frameState->urlString;
}

WebBackForwardListFrameItem* WebBackForwardListFrameItem::childItemForFrameID(FrameIdentifier frameID)
{
    if (m_frameState->frameID == frameID)
        return this;
    for (auto& child : m_children) {
        if (auto* childFrameItem = child->childItemForFrameID(frameID))
            return childFrameItem;
    }
    return nullptr;
}

RefPtr<WebBackForwardListFrameItem> WebBackForwardListFrameItem::protectedChildItemForFrameID(FrameIdentifier frameID)
{
    return childItemForFrameID(frameID);
}

WebBackForwardListItem* WebBackForwardListFrameItem::backForwardListItem() const
{
    return m_backForwardListItem.get();
}

RefPtr<WebBackForwardListItem> WebBackForwardListFrameItem::protectedBackForwardListItem() const
{
    return m_backForwardListItem.get();
}

void WebBackForwardListFrameItem::setChild(Ref<FrameState>&& frameState)
{
    ASSERT(m_backForwardListItem);
    Ref childItem = WebBackForwardListFrameItem::create(*protectedBackForwardListItem(), this, WTFMove(frameState));
    for (size_t i = 0; i < m_children.size(); i++) {
        if (m_children[i]->frameID() == childItem->m_frameState->frameID) {
            m_children[i] = WTFMove(childItem);
            return;
        }
    }
    m_children.append(WTFMove(childItem));
}

WebBackForwardListFrameItem& WebBackForwardListFrameItem::rootFrame()
{
    Ref rootFrame = *this;
    while (rootFrame->m_parent && rootFrame->m_parent->identifier().processIdentifier() == identifier().processIdentifier())
        rootFrame = *rootFrame->m_parent;
    return rootFrame.get();
}

WebBackForwardListFrameItem& WebBackForwardListFrameItem::mainFrame()
{
    Ref mainFrame = *this;
    while (mainFrame->m_parent)
        mainFrame = *mainFrame->m_parent;
    return mainFrame.get();
}

Ref<WebBackForwardListFrameItem> WebBackForwardListFrameItem::protectedMainFrame()
{
    return mainFrame();
}

void WebBackForwardListFrameItem::setWasRestoredFromSession()
{
    m_frameState->wasRestoredFromSession = true;
    for (auto& child : m_children)
        child->setWasRestoredFromSession();
}

void WebBackForwardListFrameItem::setFrameState(Ref<FrameState>&& frameState)
{
    m_frameState = WTFMove(frameState);
    m_frameState->children.clear();
}

Ref<FrameState> WebBackForwardListFrameItem::copyFrameStateWithChildren()
{
    Ref frameState = protectedFrameState()->copy();
    ASSERT(frameState->children.isEmpty());
    for (auto& child : m_children)
        frameState->children.append(child->copyFrameStateWithChildren());
    return frameState;
}

bool WebBackForwardListFrameItem::sharesAncestor(WebBackForwardListFrameItem& frameItem) const
{
    HashSet<WebCore::BackForwardFrameItemIdentifier> currentAncestors;
    for (RefPtr currentAncestor = m_parent.get(); currentAncestor; currentAncestor = currentAncestor->m_parent.get())
        currentAncestors.add(currentAncestor->m_identifier);

    for (RefPtr frameItemAncestor = frameItem.m_parent.get(); frameItemAncestor; frameItemAncestor = frameItemAncestor->m_parent.get()) {
        if (currentAncestors.contains(frameItemAncestor->m_identifier))
            return true;
    }
    return false;
}

} // namespace WebKit
