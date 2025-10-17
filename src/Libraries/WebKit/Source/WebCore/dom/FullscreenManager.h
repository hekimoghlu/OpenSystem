/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 12, 2023.
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

#if ENABLE(FULLSCREEN_API)

#include "Document.h"
#include "FrameDestructionObserverInlines.h"
#include "GCReachableRef.h"
#include "HTMLMediaElement.h"
#include "HTMLMediaElementEnums.h"
#include "LayoutRect.h"
#include "Page.h"
#include <wtf/Deque.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class DeferredPromise;
class RenderStyle;

class FullscreenManager final : public CanMakeWeakPtr<FullscreenManager>, public CanMakeCheckedPtr<FullscreenManager> {
    WTF_MAKE_TZONE_ALLOCATED(FullscreenManager);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(FullscreenManager);
public:
    FullscreenManager(Document&);
    ~FullscreenManager();

    Document& document() { return m_document.get(); }
    const Document& document() const { return m_document.get(); }
    Ref<Document> protectedDocument() const { return m_document.get(); }
    Page* page() const { return document().page(); }
    LocalFrame* frame() const { return document().frame(); }
    Element* documentElement() const { return document().documentElement(); }
    bool isSimpleFullscreenDocument() const;
    Document::BackForwardCacheState backForwardCacheState() const { return document().backForwardCacheState(); }

    // WHATWG Fullscreen API
    WEBCORE_EXPORT Element* fullscreenElement() const;
    RefPtr<Element> protectedFullscreenElement() const { return fullscreenElement(); }
    WEBCORE_EXPORT bool isFullscreenEnabled() const;
    WEBCORE_EXPORT void exitFullscreen(RefPtr<DeferredPromise>&&);

    // Mozilla versions.
    bool isFullscreen() const { return m_fullscreenElement.get(); }
    bool isFullscreenKeyboardInputAllowed() const { return m_fullscreenElement.get() && m_areKeysEnabledInFullscreen; }
    Element* currentFullscreenElement() const { return m_fullscreenElement.get(); }
    RefPtr<Element> protectedCurrentFullscreenElement() const { return currentFullscreenElement(); }
    WEBCORE_EXPORT void cancelFullscreen();

    enum FullscreenCheckType {
        EnforceIFrameAllowFullscreenRequirement,
        ExemptIFrameAllowFullscreenRequirement,
    };
    WEBCORE_EXPORT void requestFullscreenForElement(Ref<Element>&&, RefPtr<DeferredPromise>&&, FullscreenCheckType, CompletionHandler<void(bool)>&& = [](bool) { }, HTMLMediaElementEnums::VideoFullscreenMode = HTMLMediaElementEnums::VideoFullscreenModeStandard);
    WEBCORE_EXPORT bool willEnterFullscreen(Element&, HTMLMediaElementEnums::VideoFullscreenMode = HTMLMediaElementEnums::VideoFullscreenModeStandard);
    WEBCORE_EXPORT bool didEnterFullscreen();
    WEBCORE_EXPORT bool willExitFullscreen();
    WEBCORE_EXPORT bool didExitFullscreen();

    void notifyAboutFullscreenChangeOrError();

    enum class ExitMode : bool { Resize, NoResize };
    void finishExitFullscreen(Document&, ExitMode);

    void exitRemovedFullscreenElement(Element&);

    WEBCORE_EXPORT bool isAnimatingFullscreen() const;
    WEBCORE_EXPORT void setAnimatingFullscreen(bool);

    void clear();
    void emptyEventQueue();

    void updatePageFullscreenStatusIfTopDocument();

protected:
    friend class Document;

    enum class EventType : bool { Change, Error };
    void dispatchFullscreenChangeOrErrorEvent(Deque<GCReachableRef<Node>>&, EventType, bool shouldNotifyMediaElement);
    void dispatchEventForNode(Node&, EventType);
    void addDocumentToFullscreenChangeEventQueue(Document&);

private:
#if !RELEASE_LOG_DISABLED
    const Logger& logger() const { return document().logger(); }
    uint64_t logIdentifier() const { return m_logIdentifier; }
    ASCIILiteral logClassName() const { return "FullscreenManager"_s; }
    WTFLogChannel& logChannel() const;
#endif

    Document* mainFrameDocument();
    RefPtr<Document> protectedMainFrameDocument();

    WeakRef<Document, WeakPtrImplWithEventTargetData> m_document;
    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_topDocument;

    RefPtr<Element> fullscreenOrPendingElement() const { return m_fullscreenElement ? m_fullscreenElement : m_pendingFullscreenElement; }

    RefPtr<DeferredPromise> m_pendingPromise;

    bool m_pendingExitFullscreen { false };
    RefPtr<Element> m_pendingFullscreenElement;
    RefPtr<Element> m_fullscreenElement;
    Deque<GCReachableRef<Node>> m_fullscreenChangeEventTargetQueue;
    Deque<GCReachableRef<Node>> m_fullscreenErrorEventTargetQueue;

    bool m_areKeysEnabledInFullscreen { false };
    bool m_isAnimatingFullscreen { false };

#if !RELEASE_LOG_DISABLED
    const uint64_t m_logIdentifier;
#endif
};

}

#endif
