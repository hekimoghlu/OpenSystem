/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 16, 2022.
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

#include "ActivityState.h"
#include "FocusOptions.h"
#include "LayoutRect.h"
#include "LocalFrame.h"
#include "Timer.h"
#include <wtf/CheckedRef.h>
#include <wtf/Forward.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class ContainerNode;
class Document;
class Element;
class FocusNavigationScope;
class Frame;
class IntRect;
class KeyboardEvent;
class Node;
class Page;
class TreeScope;

struct FocusCandidate;

class FocusController final : public CanMakeCheckedPtr<FocusController> {
    WTF_MAKE_TZONE_ALLOCATED(FocusController);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(FocusController);
public:
    explicit FocusController(Page&, OptionSet<ActivityState>);

    enum class BroadcastFocusedFrame : bool { No, Yes };
    WEBCORE_EXPORT void setFocusedFrame(Frame*, BroadcastFocusedFrame = BroadcastFocusedFrame::Yes);
    Frame* focusedFrame() const { return m_focusedFrame.get(); }
    LocalFrame* focusedLocalFrame() const { return dynamicDowncast<LocalFrame>(m_focusedFrame.get()); }
    WEBCORE_EXPORT LocalFrame* focusedOrMainFrame() const;

    WEBCORE_EXPORT bool setInitialFocus(FocusDirection, KeyboardEvent*);
    bool advanceFocus(FocusDirection, KeyboardEvent*, bool initialFocus = false);

    WEBCORE_EXPORT bool setFocusedElement(Element*, LocalFrame&, const FocusOptions& = { });

    void setActivityState(OptionSet<ActivityState>);

    WEBCORE_EXPORT void setActive(bool);
    bool isActive() const { return m_activityState.contains(ActivityState::WindowIsActive); }

    WEBCORE_EXPORT void setFocused(bool);
    bool isFocused() const { return m_activityState.contains(ActivityState::IsFocused); }

    bool contentIsVisible() const { return m_activityState.contains(ActivityState::IsVisible); }

    // These methods are used in WebCore/bindings/objc/DOM.mm.
    WEBCORE_EXPORT Element* nextFocusableElement(Node&);
    WEBCORE_EXPORT Element* previousFocusableElement(Node&);

    void setFocusedElementNeedsRepaint();
    Seconds timeSinceFocusWasSet() const;

    WEBCORE_EXPORT bool relinquishFocusToChrome(FocusDirection);

private:
    void setActiveInternal(bool);
    void setFocusedInternal(bool);
    void setIsVisibleAndActiveInternal(bool);

    bool advanceFocusDirectionally(FocusDirection, KeyboardEvent*);
    bool advanceFocusInDocumentOrder(FocusDirection, KeyboardEvent*, bool initialFocus);

    Element* findFocusableElementAcrossFocusScope(FocusDirection, const FocusNavigationScope& startScope, Node* start, KeyboardEvent*);

    Element* findFocusableElementWithinScope(FocusDirection, const FocusNavigationScope&, Node* start, KeyboardEvent*);
    Element* nextFocusableElementWithinScope(const FocusNavigationScope&, Node* start, KeyboardEvent*);
    Element* previousFocusableElementWithinScope(const FocusNavigationScope&, Node* start, KeyboardEvent*);

    Element* findFocusableElementDescendingIntoSubframes(FocusDirection, Element*, KeyboardEvent*);

    // Searches through the given tree scope, starting from start node, for the next/previous selectable element that comes after/before start node.
    // The order followed is as specified in section 17.11.1 of the HTML4 spec, which is elements with tab indexes
    // first (from lowest to highest), and then elements without tab indexes (in document order).
    //
    // @param start The node from which to start searching. The node after this will be focused. May be null.
    //
    // @return The focus node that comes after/before start node.
    //
    // See http://www.w3.org/TR/html4/interact/forms.html#h-17.11.1
    Element* findFocusableElementOrScopeOwner(FocusDirection, const FocusNavigationScope&, Node* start, KeyboardEvent*);

    Element* findElementWithExactTabIndex(const FocusNavigationScope&, Node* start, int tabIndex, KeyboardEvent*, FocusDirection);
    
    Element* nextFocusableElementOrScopeOwner(const FocusNavigationScope&, Node* start, KeyboardEvent*);
    Element* previousFocusableElementOrScopeOwner(const FocusNavigationScope&, Node* start, KeyboardEvent*);

    bool advanceFocusDirectionallyInContainer(Node* container, const LayoutRect& startingRect, FocusDirection, KeyboardEvent*);
    void findFocusCandidateInContainer(Node& container, const LayoutRect& startingRect, FocusDirection, KeyboardEvent*, FocusCandidate& closest);

    void focusRepaintTimerFired();
    Ref<Page> protectedPage() const;

    WeakRef<Page> m_page;
    WeakPtr<Frame> m_focusedFrame;
    bool m_isChangingFocusedFrame;
    OptionSet<ActivityState> m_activityState;

    Timer m_focusRepaintTimer;
    MonotonicTime m_focusSetTime;
};

} // namespace WebCore
