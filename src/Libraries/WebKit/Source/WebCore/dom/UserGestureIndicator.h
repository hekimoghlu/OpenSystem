/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 14, 2025.
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

#include "DOMPasteAccess.h"
#include <wtf/Function.h>
#include <wtf/MonotonicTime.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UUID.h>
#include <wtf/Vector.h>
#include <wtf/WeakHashSet.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Document;
class WeakPtrImplWithEventTargetData;

enum class IsProcessingUserGesture : uint8_t { No, Yes, Potentially };

enum class CanRequestDOMPaste : bool { No, Yes };
enum class UserGestureType : uint8_t { EscapeKey, ActivationTriggering, Other };

class UserGestureToken : public RefCountedAndCanMakeWeakPtr<UserGestureToken> {
public:
    static constexpr Seconds maximumIntervalForUserGestureForwarding { 1_s }; // One second matches Gecko.
    static const Seconds& maximumIntervalForUserGestureForwardingForFetch();
    WEBCORE_EXPORT static void setMaximumIntervalForUserGestureForwardingForFetchForTesting(Seconds);

    static Ref<UserGestureToken> create(IsProcessingUserGesture isProcessingUserGesture, UserGestureType gestureType, Document* document = nullptr, std::optional<WTF::UUID> authorizationToken = std::nullopt, CanRequestDOMPaste canRequestDOMPaste = CanRequestDOMPaste::Yes)
    {
        return adoptRef(*new UserGestureToken(isProcessingUserGesture, gestureType, document, authorizationToken, canRequestDOMPaste));
    }

    WEBCORE_EXPORT ~UserGestureToken();

    IsProcessingUserGesture isProcessingUserGesture() const { return m_isProcessingUserGesture; }
    bool processingUserGesture() const { return m_scope == GestureScope::All && m_isProcessingUserGesture == IsProcessingUserGesture::Yes; }
    bool processingUserGestureForMedia() const { return m_isProcessingUserGesture == IsProcessingUserGesture::Yes || m_isProcessingUserGesture == IsProcessingUserGesture::Potentially; }
    UserGestureType gestureType() const { return m_gestureType; }

    void addDestructionObserver(Function<void(UserGestureToken&)>&& observer)
    {
        m_destructionObservers.append(WTFMove(observer));
    }

    DOMPasteAccessPolicy domPasteAccessPolicy() const { return m_domPasteAccessPolicy; }
    void didRequestDOMPasteAccess(DOMPasteAccessResponse response)
    {
        switch (response) {
        case DOMPasteAccessResponse::DeniedForGesture:
            m_domPasteAccessPolicy = DOMPasteAccessPolicy::Denied;
            break;
        case DOMPasteAccessResponse::GrantedForCommand:
            break;
        case DOMPasteAccessResponse::GrantedForGesture:
            m_domPasteAccessPolicy = DOMPasteAccessPolicy::Granted;
            break;
        }
    }
    void resetDOMPasteAccess() { m_domPasteAccessPolicy = DOMPasteAccessPolicy::NotRequestedYet; }

    enum class GestureScope { All, MediaOnly };
    void setScope(GestureScope scope) { m_scope = scope; }
    void resetScope() { m_scope = GestureScope::All; }

    // Expand the following methods if more propagation sources are added later.
    enum class IsPropagatedFromFetch : bool { No, Yes };
    void setIsPropagatedFromFetch(IsPropagatedFromFetch is) { m_isPropagatedFromFetch = is; }
    void resetIsPropagatedFromFetch() { m_isPropagatedFromFetch = IsPropagatedFromFetch::No; }
    bool isPropagatedFromFetch() const { return m_isPropagatedFromFetch == IsPropagatedFromFetch::Yes; }

    bool hasExpired(Seconds expirationInterval) const
    {
        return m_startTime + expirationInterval < MonotonicTime::now();
    }

    MonotonicTime startTime() const { return m_startTime; }

    std::optional<WTF::UUID> authorizationToken() const { return m_authorizationToken; }

    bool canRequestDOMPaste() const { return m_canRequestDOMPaste == CanRequestDOMPaste::Yes; }

    bool isValidForDocument(const Document&) const;

    void forEachImpactedDocument(Function<void(Document&)>&&);

private:
    UserGestureToken(IsProcessingUserGesture, UserGestureType, Document*, std::optional<WTF::UUID> authorizationToken, CanRequestDOMPaste);

    IsProcessingUserGesture m_isProcessingUserGesture = IsProcessingUserGesture::No;
    Vector<Function<void(UserGestureToken&)>> m_destructionObservers;
    UserGestureType m_gestureType;
    WeakHashSet<Document, WeakPtrImplWithEventTargetData> m_documentsImpactedByUserGesture;
    CanRequestDOMPaste m_canRequestDOMPaste { CanRequestDOMPaste::No };
    DOMPasteAccessPolicy m_domPasteAccessPolicy { DOMPasteAccessPolicy::NotRequestedYet };
    GestureScope m_scope { GestureScope::All };
    MonotonicTime m_startTime { MonotonicTime::now() };
    IsPropagatedFromFetch m_isPropagatedFromFetch { IsPropagatedFromFetch::No };
    std::optional<WTF::UUID> m_authorizationToken;
};

class UserGestureIndicator {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(UserGestureIndicator, WEBCORE_EXPORT);
    WTF_MAKE_NONCOPYABLE(UserGestureIndicator);
public:
    WEBCORE_EXPORT static RefPtr<UserGestureToken> currentUserGesture();

    WEBCORE_EXPORT static bool processingUserGesture(const Document* = nullptr);
    WEBCORE_EXPORT static bool processingUserGestureForMedia();

    // If a document is provided, its last known user gesture timestamp is updated.
    enum class ProcessInteractionStyle { Immediate, Delayed, Never };
    WEBCORE_EXPORT explicit UserGestureIndicator(std::optional<IsProcessingUserGesture>, Document* = nullptr, UserGestureType = UserGestureType::ActivationTriggering, ProcessInteractionStyle = ProcessInteractionStyle::Immediate, std::optional<WTF::UUID> authorizationToken = std::nullopt, CanRequestDOMPaste = CanRequestDOMPaste::Yes);
    WEBCORE_EXPORT explicit UserGestureIndicator(RefPtr<UserGestureToken>, UserGestureToken::GestureScope = UserGestureToken::GestureScope::All, UserGestureToken::IsPropagatedFromFetch = UserGestureToken::IsPropagatedFromFetch::No);
    WEBCORE_EXPORT ~UserGestureIndicator();

    WEBCORE_EXPORT std::optional<WTF::UUID> authorizationToken() const;

private:
    RefPtr<UserGestureToken> m_previousToken;
};

}
