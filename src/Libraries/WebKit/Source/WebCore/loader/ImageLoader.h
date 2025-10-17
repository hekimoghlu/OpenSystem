/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 11, 2023.
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

#include "CachedImageClient.h"
#include "CachedResourceHandle.h"
#include "Element.h"
#include "LoaderMalloc.h"
#include "Timer.h"
#include <wtf/Vector.h>
#include <wtf/text/AtomString.h>

namespace WebCore {

class DeferredPromise;
class Document;
class ImageLoader;
class Page;
class RenderImageResource;
struct ImageCandidate;

template<typename T, typename Counter> class EventSender;
using ImageEventSender = EventSender<ImageLoader, SingleThreadWeakPtrImpl>;

enum class RelevantMutation : bool { No, Yes };
enum class LazyImageLoadState : uint8_t { None, Deferred, LoadImmediately, FullImage };

class ImageLoader : public CachedImageClient {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(ImageLoader);
public:
    virtual ~ImageLoader();

    void ref() const;
    void deref() const;

    // This function should be called when the element is attached to a document; starts
    // loading if a load hasn't already been started.
    void updateFromElement(RelevantMutation = RelevantMutation::No);

    // This function should be called whenever the 'src' attribute is set.
    // Starts new load unconditionally (matches Firefox and Opera behavior).
    void updateFromElementIgnoringPreviousError(RelevantMutation = RelevantMutation::No);

    void updateFromElementIgnoringPreviousErrorToSameValue();

    void elementDidMoveToNewDocument(Document&);

    Element& element() { return m_element.get(); }
    const Element& element() const { return m_element.get(); }
    Ref<Element> protectedElement() const { return m_element.get(); }

    bool shouldIgnoreCandidateWhenLoadingFromArchive(const ImageCandidate&) const;

    bool imageComplete() const { return m_imageComplete; }

    CachedImage* image() const { return m_image.get(); }
    CachedResourceHandle<CachedImage> protectedImage() const;
    void clearImage(); // Cancels pending load events, and doesn't dispatch new ones.
    
    size_t pendingDecodePromisesCountForTesting() const { return m_decodingPromises.size(); }
    void decode(Ref<DeferredPromise>&&);

    void setLoadManually(bool loadManually) { m_loadManually = loadManually; }

    // FIXME: Delete this code. beforeload event no longer exists.
    bool hasPendingBeforeLoadEvent() const { return m_hasPendingBeforeLoadEvent; }
    bool hasPendingActivity() const;

    void dispatchPendingEvent(ImageEventSender*, const AtomString& eventType);

    static void dispatchPendingLoadEvents(Page*);

    void loadDeferredImage();

    bool isDeferred() const { return m_lazyImageLoadState == LazyImageLoadState::Deferred || m_lazyImageLoadState == LazyImageLoadState::LoadImmediately; }

    Document& document() { return m_element->document(); }
    Ref<Document> protectedDocument() { return m_element->document(); }

protected:
    explicit ImageLoader(Element&);
    void notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess = LoadWillContinueInAnotherProcess::No) override;

private:
    void resetLazyImageLoading(Document&);

    virtual void dispatchLoadEvent() = 0;

    void updatedHasPendingEvent();
    void didUpdateCachedImage(RelevantMutation, CachedResourceHandle<CachedImage>&&);

    void dispatchPendingBeforeLoadEvent();
    void dispatchPendingLoadEvent();
    void dispatchPendingErrorEvent();

    RenderImageResource* renderImageResource();
    void updateRenderer();

    void clearImageWithoutConsideringPendingLoadEvent();
    void clearFailedLoadURL();

    bool hasPendingDecodePromises() const { return !m_decodingPromises.isEmpty(); }
    void resolveDecodePromises();
    void rejectDecodePromises(ASCIILiteral message);
    void decode();
    
    void timerFired();

    VisibleInViewportState imageVisibleInViewport(const Document&) const override;

    WeakRef<Element, WeakPtrImplWithEventTargetData> m_element;
    CachedResourceHandle<CachedImage> m_image;
    Timer m_derefElementTimer;
    RefPtr<Element> m_protectedElement;
    AtomString m_failedLoadURL;
    AtomString m_pendingURL;
    Vector<RefPtr<DeferredPromise>> m_decodingPromises;
    bool m_hasPendingBeforeLoadEvent : 1;
    bool m_hasPendingLoadEvent : 1;
    bool m_hasPendingErrorEvent : 1;
    bool m_imageComplete : 1;
    bool m_loadManually : 1;
    bool m_elementIsProtected : 1;
    LazyImageLoadState m_lazyImageLoadState { LazyImageLoadState::None };
};

}
