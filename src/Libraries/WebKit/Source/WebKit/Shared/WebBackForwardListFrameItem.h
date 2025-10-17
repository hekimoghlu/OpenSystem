/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 27, 2024.
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
#pragma once

#include <WebCore/BackForwardFrameItemIdentifier.h>
#include <WebCore/BackForwardItemIdentifier.h>
#include <WebCore/FrameIdentifier.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>

namespace WebKit {

class FrameState;
class WebBackForwardListItem;

class WebBackForwardListFrameItem : public RefCountedAndCanMakeWeakPtr<WebBackForwardListFrameItem> {
public:
    static Ref<WebBackForwardListFrameItem> create(WebBackForwardListItem&, WebBackForwardListFrameItem* parentItem, Ref<FrameState>&&);
    ~WebBackForwardListFrameItem();

    static WebBackForwardListFrameItem* itemForID(WebCore::BackForwardItemIdentifier, WebCore::BackForwardFrameItemIdentifier);

    FrameState& frameState() const { return m_frameState; }
    Ref<FrameState> protectedFrameState() const { return m_frameState; }
    void setFrameState(Ref<FrameState>&&);

    Ref<FrameState> copyFrameStateWithChildren();

    std::optional<WebCore::FrameIdentifier> frameID() const;
    WebCore::BackForwardFrameItemIdentifier identifier() const { return m_identifier; }
    const String& url() const;

    WebBackForwardListFrameItem* parent() const { return m_parent.get(); }
    RefPtr<WebBackForwardListFrameItem> protectedParent() const { return m_parent.get(); }
    void setParent(WebBackForwardListFrameItem* parent) { m_parent = parent; }
    bool sharesAncestor(WebBackForwardListFrameItem&) const;

    WebBackForwardListFrameItem& rootFrame();
    WebBackForwardListFrameItem& mainFrame();
    Ref<WebBackForwardListFrameItem> protectedMainFrame();
    WebBackForwardListFrameItem* childItemForFrameID(WebCore::FrameIdentifier);
    RefPtr<WebBackForwardListFrameItem> protectedChildItemForFrameID(WebCore::FrameIdentifier);

    WebBackForwardListItem* backForwardListItem() const;
    RefPtr<WebBackForwardListItem> protectedBackForwardListItem() const;

    void setChild(Ref<FrameState>&&);
    void clearChildren() { m_children.clear(); }

    void setWasRestoredFromSession();

private:
    WebBackForwardListFrameItem(WebBackForwardListItem&, WebBackForwardListFrameItem* parentItem, Ref<FrameState>&&);

    static HashMap<std::pair<WebCore::BackForwardFrameItemIdentifier, WebCore::BackForwardItemIdentifier>, WeakRef<WebBackForwardListFrameItem>>& allItems();

    WeakPtr<WebBackForwardListItem> m_backForwardListItem;
    const WebCore::BackForwardFrameItemIdentifier m_identifier;
    Ref<FrameState> m_frameState;
    WeakPtr<WebBackForwardListFrameItem> m_parent;
    Vector<Ref<WebBackForwardListFrameItem>> m_children;
};

} // namespace WebKit
