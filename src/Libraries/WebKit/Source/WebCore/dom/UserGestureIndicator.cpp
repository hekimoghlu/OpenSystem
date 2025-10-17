/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 18, 2025.
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
#include "UserGestureIndicator.h"

#include "Document.h"
#include "FrameDestructionObserverInlines.h"
#include "LocalDOMWindow.h"
#include "LocalFrame.h"
#include "Logging.h"
#include "Page.h"
#include "ResourceLoadObserver.h"
#include "SecurityOrigin.h"
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(UserGestureIndicator);

static RefPtr<UserGestureToken>& currentToken()
{
    ASSERT(isMainThread());
    static NeverDestroyed<RefPtr<UserGestureToken>> token;
    return token;
}

UserGestureToken::UserGestureToken(IsProcessingUserGesture isProcessingUserGesture, UserGestureType gestureType, Document* document, std::optional<WTF::UUID> authorizationToken, CanRequestDOMPaste canRequestDOMPaste)
    : m_isProcessingUserGesture(isProcessingUserGesture)
    , m_gestureType(gestureType)
    , m_canRequestDOMPaste(canRequestDOMPaste)
    , m_authorizationToken(authorizationToken)
{
    if (!document || !processingUserGesture())
        return;

    // User gesture is valid for the document that received the user gesture, all of its ancestors
    // as well as all same-origin documents on the page.
    m_documentsImpactedByUserGesture.add(*document);

    RefPtr documentFrame = document->frame();
    if (!documentFrame)
        return;

    for (RefPtr ancestorFrame = documentFrame->tree().parent(); ancestorFrame; ancestorFrame = ancestorFrame->tree().parent()) {
        RefPtr localAncestor = dynamicDowncast<LocalFrame>(ancestorFrame);
        if (!localAncestor)
            continue;
        if (RefPtr ancestorDocument = localAncestor->document())
            m_documentsImpactedByUserGesture.add(*ancestorDocument);
    }

    Ref documentOrigin = document->securityOrigin();
    for (RefPtr frame = &documentFrame->tree().top(); frame; frame = frame->tree().traverseNext()) {
        RefPtr localFrame = dynamicDowncast<LocalFrame>(frame);
        if (!localFrame)
            continue;
        RefPtr frameDocument = localFrame->document();
        Ref frameOrigin = frameDocument->securityOrigin();
        if (frameDocument && documentOrigin->isSameOriginDomain(frameOrigin.get()))
            m_documentsImpactedByUserGesture.add(*frameDocument);
    }
}

UserGestureToken::~UserGestureToken()
{
    for (auto& observer : m_destructionObservers)
        observer(*this);
}

static Seconds maxIntervalForUserGestureForwardingForFetch { 10 };
const Seconds& UserGestureToken::maximumIntervalForUserGestureForwardingForFetch()
{
    return maxIntervalForUserGestureForwardingForFetch;
}

void UserGestureToken::setMaximumIntervalForUserGestureForwardingForFetchForTesting(Seconds value)
{
    maxIntervalForUserGestureForwardingForFetch = WTFMove(value);
}

bool UserGestureToken::isValidForDocument(const Document& document) const
{
    return m_documentsImpactedByUserGesture.contains(document);
}

void UserGestureToken::forEachImpactedDocument(Function<void(Document&)>&& function)
{
    m_documentsImpactedByUserGesture.forEach(function);
}

UserGestureIndicator::UserGestureIndicator(std::optional<IsProcessingUserGesture> isProcessingUserGesture, Document* document, UserGestureType gestureType, ProcessInteractionStyle processInteractionStyle, std::optional<WTF::UUID> authorizationToken, CanRequestDOMPaste canRequestDOMPaste)
    : m_previousToken { currentToken() }
{
    ASSERT(isMainThread());

    if (isProcessingUserGesture)
        currentToken() = UserGestureToken::create(isProcessingUserGesture.value(), gestureType, document, authorizationToken, canRequestDOMPaste);

    if (isProcessingUserGesture && document && currentToken()->processingUserGesture()) {
        document->updateLastHandledUserGestureTimestamp(currentToken()->startTime());
        if (processInteractionStyle == ProcessInteractionStyle::Immediate) {
            RefPtr mainFrameDocument = document->protectedMainFrameDocument();
            if (mainFrameDocument)
                ResourceLoadObserver::shared().logUserInteractionWithReducedTimeResolution(*mainFrameDocument);
            else
                LOG_ONCE(SiteIsolation, "Unable to properly construct UserGestureIndicator::UserGestureIndicator() without access to the main frame document ");
        }
        if (RefPtr page = document->protectedPage())
            page->setUserDidInteractWithPage(true);
        if (RefPtr frame = document->frame(); frame && !frame->hasHadUserInteraction()) {
            for (RefPtr<Frame> ancestor = WTFMove(frame); ancestor; ancestor = ancestor->tree().parent()) {
                if (RefPtr localAncestor = dynamicDowncast<LocalFrame>(ancestor)) {
                    localAncestor->setHasHadUserInteraction();
                    if (RefPtr ancestorDocument = localAncestor->protectedDocument())
                        ancestorDocument->updateLastHandledUserGestureTimestamp(currentToken()->startTime());
                }
            }
        }

        // https://html.spec.whatwg.org/multipage/interaction.html#user-activation-processing-model
        // When a user interaction causes firing of an activation triggering input event in a Document...
        // NOTE: Only activate the relevent DOMWindow when the gestureType is an ActivationTriggering one
        RefPtr window = document->domWindow();
        if (window && gestureType == UserGestureType::ActivationTriggering)
            window->notifyActivated(currentToken()->startTime());
    }
}

UserGestureIndicator::UserGestureIndicator(RefPtr<UserGestureToken> token, UserGestureToken::GestureScope scope, UserGestureToken::IsPropagatedFromFetch isPropagatedFromFetch)
{
    // Silently ignore UserGestureIndicators on non main threads.
    if (!isMainThread())
        return;

    // It is only safe to use currentToken() on the main thread.
    m_previousToken = currentToken();

    if (token) {
        token->setScope(scope);
        token->setIsPropagatedFromFetch(isPropagatedFromFetch);
        currentToken() = token;
    }
}

UserGestureIndicator::~UserGestureIndicator()
{
    if (!isMainThread())
        return;
    
    if (auto token = currentToken()) {
        token->resetDOMPasteAccess();
        token->resetScope();
        token->resetIsPropagatedFromFetch();
    }

    currentToken() = m_previousToken;
}

RefPtr<UserGestureToken> UserGestureIndicator::currentUserGesture()
{
    if (!isMainThread())
        return nullptr;

    return currentToken();
}

bool UserGestureIndicator::processingUserGesture(const Document* document)
{
    if (!isMainThread())
        return false;

    RefPtr token = currentToken();
    if (!token || !token->processingUserGesture())
        return false;

    return !document || token->isValidForDocument(*document);
}

bool UserGestureIndicator::processingUserGestureForMedia()
{
    if (!isMainThread())
        return false;

    RefPtr token = currentToken();
    return token ? token->processingUserGestureForMedia() : false;
}

std::optional<WTF::UUID> UserGestureIndicator::authorizationToken() const
{
    if (!isMainThread())
        return std::nullopt;

    RefPtr token = currentToken();
    return token ? token->authorizationToken() : std::nullopt;
}

}
