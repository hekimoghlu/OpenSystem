/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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

#include "BackForwardItemIdentifier.h"
#include "CrossOriginOpenerPolicy.h"
#include "FrameLoaderTypes.h"
#include "GlobalFrameIdentifier.h"
#include "LayoutPoint.h"
#include "NavigationRequester.h"
#include "PrivateClickMeasurement.h"
#include "ResourceRequest.h"
#include "SecurityOrigin.h"
#include "UserGestureIndicator.h"
#include <wtf/Forward.h>

namespace WebCore {

class Document;
class Event;
class HistoryItem;
class MouseEvent;
class UIEventWithKeyState;

enum class SyntheticClickType : uint8_t;
enum class MouseButton : int8_t;
enum class NavigationNavigationType : uint8_t;

// NavigationAction should never hold a strong reference to the originating document either directly
// or indirectly as doing so prevents its destruction even after navigating away from it because
// DocumentLoader keeps around the NavigationAction for the last navigation.
class NavigationAction {
public:
    NavigationAction();
    WEBCORE_EXPORT NavigationAction(Document&, const ResourceRequest&, InitiatedByMainFrame, bool, NavigationType = NavigationType::Other, ShouldOpenExternalURLsPolicy = ShouldOpenExternalURLsPolicy::ShouldNotAllow, Event* = nullptr, const AtomString& downloadAttribute = nullAtom());
    NavigationAction(Document&, const ResourceRequest&, InitiatedByMainFrame, bool, FrameLoadType, bool isFormSubmission, Event* = nullptr, ShouldOpenExternalURLsPolicy = ShouldOpenExternalURLsPolicy::ShouldNotAllow, const AtomString& downloadAttribute = nullAtom());

    WEBCORE_EXPORT ~NavigationAction();

    WEBCORE_EXPORT NavigationAction(const NavigationAction&);
    NavigationAction& operator=(const NavigationAction&);

    NavigationAction(NavigationAction&&);
    NavigationAction& operator=(NavigationAction&&);

    const std::optional<NavigationRequester>& requester() const { return m_requester; }

    struct UIEventWithKeyStateData {
        UIEventWithKeyStateData(const UIEventWithKeyState&);

        bool isTrusted;
        bool shiftKey;
        bool ctrlKey;
        bool altKey;
        bool metaKey;
    };
    struct MouseEventData : UIEventWithKeyStateData {
        MouseEventData(const MouseEvent&);

        LayoutPoint absoluteLocation;
        FloatPoint locationInRootViewCoordinates;
        MouseButton button;
        SyntheticClickType syntheticClickType;
        bool buttonDown;
    };
    const std::optional<UIEventWithKeyStateData>& keyStateEventData() const { return m_keyStateEventData; }
    const std::optional<MouseEventData>& mouseEventData() const { return m_mouseEventData; }

    NavigationAction copyWithShouldOpenExternalURLsPolicy(ShouldOpenExternalURLsPolicy) const;

    bool isEmpty() const { return !m_requester || m_requester->url.isEmpty() || m_originalRequest.url().isEmpty(); }

    URL url() const { return m_originalRequest.url(); }
    const ResourceRequest& originalRequest() const { return m_originalRequest; }

    NavigationType type() const { return m_type; }

    bool processingUserGesture() const { return m_userGestureToken ? m_userGestureToken->processingUserGesture() : false; }
    RefPtr<UserGestureToken> userGestureToken() const { return m_userGestureToken; }

    ShouldOpenExternalURLsPolicy shouldOpenExternalURLsPolicy() const { return m_shouldOpenExternalURLsPolicy; }
    void setShouldOpenExternalURLsPolicy(ShouldOpenExternalURLsPolicy policy) {  m_shouldOpenExternalURLsPolicy = policy; }
    InitiatedByMainFrame initiatedByMainFrame() const { return m_initiatedByMainFrame; }

    const AtomString& downloadAttribute() const { return m_downloadAttribute; }

    bool treatAsSameOriginNavigation() const { return m_treatAsSameOriginNavigation; }

    bool hasOpenedFrames() const { return m_hasOpenedFrames; }
    void setHasOpenedFrames(bool value) { m_hasOpenedFrames = value; }

    bool openedByDOMWithOpener() const { return m_openedByDOMWithOpener; }
    void setOpenedByDOMWithOpener() { m_openedByDOMWithOpener = true; }

    NewFrameOpenerPolicy newFrameOpenerPolicy() const { return m_newFrameOpenerPolicy; }
    void setNewFrameOpenerPolicy(NewFrameOpenerPolicy newFrameOpenerPolicy) { m_newFrameOpenerPolicy = newFrameOpenerPolicy; }

    void setTargetBackForwardItem(HistoryItem&);
    const std::optional<BackForwardItemIdentifier>& targetBackForwardItemIdentifier() const { return m_targetBackForwardItemIdentifier; }

    void setSourceBackForwardItem(HistoryItem*);
    const std::optional<BackForwardItemIdentifier>& sourceBackForwardItemIdentifier() const { return m_sourceBackForwardItemIdentifier; }

    LockHistory lockHistory() const { return m_lockHistory; }
    void setLockHistory(LockHistory lockHistory) { m_lockHistory = lockHistory; }

    LockBackForwardList lockBackForwardList() const { return m_lockBackForwardList; }
    void setLockBackForwardList(LockBackForwardList lockBackForwardList) { m_lockBackForwardList = lockBackForwardList; }

    const std::optional<PrivateClickMeasurement>& privateClickMeasurement() const { return m_privateClickMeasurement; };
    void setPrivateClickMeasurement(PrivateClickMeasurement&& privateClickMeasurement) { m_privateClickMeasurement = privateClickMeasurement; };

    // The shouldReplaceDocumentIfJavaScriptURL parameter will go away when the FIXME to eliminate the
    // corresponding parameter from ScriptController::executeIfJavaScriptURL() is addressed.
    ShouldReplaceDocumentIfJavaScriptURL shouldReplaceDocumentIfJavaScriptURL() const { return m_shouldReplaceDocumentIfJavaScriptURL; }
    void setShouldReplaceDocumentIfJavaScriptURL(ShouldReplaceDocumentIfJavaScriptURL shouldReplaceDocumentIfJavaScriptURL) { m_shouldReplaceDocumentIfJavaScriptURL = shouldReplaceDocumentIfJavaScriptURL; }

    bool isRequestFromClientOrUserInput() const { return m_isRequestFromClientOrUserInput; }
    void setIsRequestFromClientOrUserInput(bool isRequestFromClientOrUserInput) { m_isRequestFromClientOrUserInput = isRequestFromClientOrUserInput; }

    bool isInitialFrameSrcLoad() const { return m_isInitialFrameSrcLoad; }
    void setIsInitialFrameSrcLoad(bool isInitialFrameSrcLoad) { m_isInitialFrameSrcLoad = isInitialFrameSrcLoad; }

    std::optional<NavigationNavigationType> navigationAPIType() const { return m_navigationAPIType; }
    void setNavigationAPIType(NavigationNavigationType navigationAPIType) { m_navigationAPIType = navigationAPIType; }

    bool isFromNavigationAPI() const { return m_isFromNavigationAPI; }
    void setIsFromNavigationAPI(bool isFromNavigationAPI) { m_isFromNavigationAPI = isFromNavigationAPI; }

private:
    // Do not add a strong reference to the originating document or a subobject that holds the
    // originating document. See comment above the class for more details.
    std::optional<NavigationRequester> m_requester;
    ResourceRequest m_originalRequest;
    std::optional<UIEventWithKeyStateData> m_keyStateEventData;
    std::optional<MouseEventData> m_mouseEventData;
    RefPtr<UserGestureToken> m_userGestureToken { UserGestureIndicator::currentUserGesture() };
    AtomString m_downloadAttribute;
    std::optional<BackForwardItemIdentifier> m_targetBackForwardItemIdentifier;
    std::optional<BackForwardItemIdentifier> m_sourceBackForwardItemIdentifier;
    std::optional<PrivateClickMeasurement> m_privateClickMeasurement;
    ShouldReplaceDocumentIfJavaScriptURL m_shouldReplaceDocumentIfJavaScriptURL { ReplaceDocumentIfJavaScriptURL };

    NavigationType m_type;
    std::optional<NavigationNavigationType> m_navigationAPIType;
    ShouldOpenExternalURLsPolicy m_shouldOpenExternalURLsPolicy;
    InitiatedByMainFrame m_initiatedByMainFrame;

    bool m_treatAsSameOriginNavigation { false };
    bool m_hasOpenedFrames { false };
    bool m_openedByDOMWithOpener { false };
    bool m_isRequestFromClientOrUserInput { false };
    bool m_isInitialFrameSrcLoad { false };
    LockHistory m_lockHistory { LockHistory::No };
    LockBackForwardList m_lockBackForwardList { LockBackForwardList::No };
    NewFrameOpenerPolicy m_newFrameOpenerPolicy { NewFrameOpenerPolicy::Allow };
    bool m_isFromNavigationAPI { false };
};

} // namespace WebCore
