/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 30, 2023.
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

#include "AXRemoteFrame.h"
#include "AccessibilityObject.h"
#include "ScrollView.h"

namespace WebCore {
    
class AXRemoteFrame;
class AccessibilityScrollbar;
class Scrollbar;
class ScrollView;
    
class AccessibilityScrollView final : public AccessibilityObject {
public:
    static Ref<AccessibilityScrollView> create(AXID, ScrollView&);
    AccessibilityRole determineAccessibilityRole() final { return AccessibilityRole::ScrollArea; }
    ScrollView* scrollView() const final { return currentScrollView(); }

    virtual ~AccessibilityScrollView();

    AccessibilityObject* webAreaObject() const final;
    void setNeedsToUpdateChildren() final { m_childrenDirty = true; }

    RefPtr<AXRemoteFrame> remoteFrame() const { return m_remoteFrame; }

private:
    explicit AccessibilityScrollView(AXID, ScrollView&);
    void detachRemoteParts(AccessibilityDetachmentType) final;

    ScrollView* currentScrollView() const;
    ScrollableArea* getScrollableAreaIfScrollable() const final { return currentScrollView(); }
    void scrollTo(const IntPoint&) const final;
    bool computeIsIgnored() const final;
    bool isAccessibilityScrollViewInstance() const final { return true; }
    bool isEnabled() const final { return true; }
    bool hasRemoteFrameChild() const final { return m_remoteFrame; }

    bool isAttachment() const final;
    PlatformWidget platformWidget() const final;
    Widget* widgetForAttachmentView() const final { return currentScrollView(); }

    AccessibilityObject* scrollBar(AccessibilityOrientation) final;
    void addChildren() final;
    void clearChildren() final;
    AccessibilityObject* accessibilityHitTest(const IntPoint&) const final;
    void updateChildrenIfNecessary() final;
    void updateScrollbars();
    void setFocused(bool) final;
    bool canSetFocusAttribute() const final;
    bool isFocused() const final;
    void addRemoteFrameChild();

    Document* document() const final;
    LocalFrameView* documentFrameView() const final;
    LayoutRect elementRect() const final;
    AccessibilityObject* parentObject() const final;

    AccessibilityObject* firstChild() const final { return webAreaObject(); }
    AccessibilityScrollbar* addChildScrollbar(Scrollbar*);
    void removeChildScrollbar(AccessibilityObject*);

    SingleThreadWeakPtr<ScrollView> m_scrollView;
    WeakPtr<HTMLFrameOwnerElement, WeakPtrImplWithEventTargetData> m_frameOwnerElement;
    RefPtr<AccessibilityObject> m_horizontalScrollbar;
    RefPtr<AccessibilityObject> m_verticalScrollbar;
    bool m_childrenDirty;
    RefPtr<AXRemoteFrame> m_remoteFrame;
};
    
} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::AccessibilityScrollView) \
    static bool isType(const WebCore::AccessibilityObject& object) { return object.isAccessibilityScrollViewInstance(); } \
SPECIALIZE_TYPE_TRAITS_END()
