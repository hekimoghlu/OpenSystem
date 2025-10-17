/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 23, 2022.
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
#include "NavigationAction.h"

#include "DocumentInlines.h"
#include "FrameLoader.h"
#include "HistoryItem.h"
#include "LocalFrame.h"
#include "MouseEvent.h"
#include "OriginAccessPatterns.h"

namespace WebCore {

NavigationAction::UIEventWithKeyStateData::UIEventWithKeyStateData(const UIEventWithKeyState& uiEvent)
    : isTrusted { uiEvent.isTrusted() }
    , shiftKey { uiEvent.shiftKey() }
    , ctrlKey { uiEvent.ctrlKey() }
    , altKey { uiEvent.altKey() }
    , metaKey { uiEvent.metaKey() }
{
}

NavigationAction::MouseEventData::MouseEventData(const MouseEvent& mouseEvent)
    : UIEventWithKeyStateData { mouseEvent }
    , absoluteLocation { mouseEvent.absoluteLocation() }
    , locationInRootViewCoordinates { mouseEvent.locationInRootViewCoordinates() }
    , button { mouseEvent.button() }
    , syntheticClickType { mouseEvent.syntheticClickType() }
    , buttonDown { mouseEvent.buttonDown() }
{
}

NavigationAction::NavigationAction() = default;
NavigationAction::~NavigationAction() = default;

NavigationAction::NavigationAction(const NavigationAction&) = default;
NavigationAction::NavigationAction(NavigationAction&&) = default;

NavigationAction& NavigationAction::operator=(const NavigationAction&) = default;
NavigationAction& NavigationAction::operator=(NavigationAction&&) = default;

static bool shouldTreatAsSameOriginNavigation(const Document& document, const URL& url)
{
    return url.protocolIsAbout() || url.protocolIsData() || (url.protocolIsBlob() && document.protectedSecurityOrigin()->canRequest(url, OriginAccessPatternsForWebProcess::singleton()));
}

static std::optional<NavigationAction::UIEventWithKeyStateData> keyStateDataForFirstEventWithKeyState(Event* event)
{
    if (UIEventWithKeyState* uiEvent = findEventWithKeyState(event))
        return NavigationAction::UIEventWithKeyStateData { *uiEvent };
    return std::nullopt;
}

static std::optional<NavigationAction::MouseEventData> mouseEventDataForFirstMouseEvent(Event* event)
{
    for (auto* e = event; e; e = e->underlyingEvent()) {
        if (auto* mouseEvent = dynamicDowncast<MouseEvent>(e))
            return NavigationAction::MouseEventData { *mouseEvent };
    }
    return { };
}

static NavigationType navigationType(FrameLoadType frameLoadType, bool isFormSubmission, bool haveEvent)
{
    if (isFormSubmission)
        return NavigationType::FormSubmitted;
    if (haveEvent)
        return NavigationType::LinkClicked;
    if (isReload(frameLoadType))
        return NavigationType::Reload;
    if (isBackForwardLoadType(frameLoadType))
        return NavigationType::BackForward;
    return NavigationType::Other;
}

NavigationAction::NavigationAction(Document& requester, const ResourceRequest& originalRequest, InitiatedByMainFrame initiatedByMainFrame, bool isRequestFromClientOrUserInput, NavigationType type, ShouldOpenExternalURLsPolicy shouldOpenExternalURLsPolicy, Event* event, const AtomString& downloadAttribute)
    : m_requester { NavigationRequester::from(requester) }
    , m_originalRequest { originalRequest }
    , m_keyStateEventData { keyStateDataForFirstEventWithKeyState(event) }
    , m_mouseEventData { mouseEventDataForFirstMouseEvent(event) }
    , m_downloadAttribute { downloadAttribute }
    , m_type { type }
    , m_shouldOpenExternalURLsPolicy { shouldOpenExternalURLsPolicy }
    , m_initiatedByMainFrame { initiatedByMainFrame }
    , m_treatAsSameOriginNavigation { shouldTreatAsSameOriginNavigation(requester, originalRequest.url()) }
    , m_isRequestFromClientOrUserInput { isRequestFromClientOrUserInput }
{
}

NavigationAction::NavigationAction(Document& requester, const ResourceRequest& originalRequest, InitiatedByMainFrame initiatedByMainFrame, bool isRequestFromClientOrUserInput, FrameLoadType frameLoadType, bool isFormSubmission, Event* event, ShouldOpenExternalURLsPolicy shouldOpenExternalURLsPolicy, const AtomString& downloadAttribute)
    : m_requester { NavigationRequester::from(requester) }
    , m_originalRequest { originalRequest }
    , m_keyStateEventData { keyStateDataForFirstEventWithKeyState(event) }
    , m_mouseEventData { mouseEventDataForFirstMouseEvent(event) }
    , m_downloadAttribute { downloadAttribute }
    , m_type { navigationType(frameLoadType, isFormSubmission, !!event) }
    , m_shouldOpenExternalURLsPolicy { shouldOpenExternalURLsPolicy }
    , m_initiatedByMainFrame { initiatedByMainFrame }
    , m_treatAsSameOriginNavigation { shouldTreatAsSameOriginNavigation(requester, originalRequest.url()) }
    , m_isRequestFromClientOrUserInput { isRequestFromClientOrUserInput }
{
}

NavigationAction NavigationAction::copyWithShouldOpenExternalURLsPolicy(ShouldOpenExternalURLsPolicy shouldOpenExternalURLsPolicy) const
{
    NavigationAction result(*this);
    result.m_shouldOpenExternalURLsPolicy = shouldOpenExternalURLsPolicy;
    return result;
}

void NavigationAction::setTargetBackForwardItem(HistoryItem& item)
{
    m_targetBackForwardItemIdentifier = item.itemID();
}

void NavigationAction::setSourceBackForwardItem(HistoryItem* item)
{
    m_sourceBackForwardItemIdentifier = item ? std::make_optional(item->itemID()) : std::nullopt;
}

}
