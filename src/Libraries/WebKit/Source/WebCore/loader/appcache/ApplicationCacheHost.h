/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 2, 2021.
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

#include "LoaderMalloc.h"
#include <wtf/CheckedRef.h>
#include <wtf/Deque.h>
#include <wtf/URL.h>
#include <wtf/Vector.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class ApplicationCache;
class ApplicationCacheGroup;
class ApplicationCacheResource;
class ApplicationCacheStorage;
class SharedBuffer;
class DOMApplicationCache;
class DocumentLoader;
class LocalFrame;
class ResourceError;
class ResourceLoader;
class ResourceRequest;
class ResourceResponse;
class SubstituteData;
class WeakPtrImplWithEventTargetData;

class ApplicationCacheHost {
    WTF_MAKE_NONCOPYABLE(ApplicationCacheHost); WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
public:
    // The Status numeric values are specified in the HTML5 spec.
    enum Status {
        UNCACHED = 0,
        IDLE = 1,
        CHECKING = 2,
        DOWNLOADING = 3,
        UPDATEREADY = 4,
        OBSOLETE = 5
    };

    struct CacheInfo {
        URL manifest;
        double creationTime;
        double updateTime;
        long long size;
    };

    struct ResourceInfo {
        URL resource;
        bool isMaster;
        bool isManifest;
        bool isFallback;
        bool isForeign;
        bool isExplicit;
        long long size;
    };

    explicit ApplicationCacheHost(DocumentLoader&);
    ~ApplicationCacheHost();

    static URL createFileURL(const String&);

    void selectCacheWithoutManifest();
    void selectCacheWithManifest(const URL& manifestURL);

    bool canLoadMainResource(const ResourceRequest&);

    void maybeLoadMainResource(const ResourceRequest&, SubstituteData&);
    void maybeLoadMainResourceForRedirect(const ResourceRequest&, SubstituteData&);
    bool maybeLoadFallbackForMainResponse(const ResourceRequest&, const ResourceResponse&);
    void mainResourceDataReceived(const SharedBuffer&, long long encodedDataLength, bool allAtOnce);
    void finishedLoadingMainResource();
    void failedLoadingMainResource();

    WEBCORE_EXPORT bool maybeLoadResource(ResourceLoader&, const ResourceRequest&, const URL& originalURL);
    WEBCORE_EXPORT bool maybeLoadFallbackForRedirect(ResourceLoader*, ResourceRequest&, const ResourceResponse&);
    WEBCORE_EXPORT bool maybeLoadFallbackForResponse(ResourceLoader*, const ResourceResponse&);
    WEBCORE_EXPORT bool maybeLoadFallbackForError(ResourceLoader*, const ResourceError&);

    bool maybeLoadSynchronously(ResourceRequest&, ResourceError&, ResourceResponse&, RefPtr<SharedBuffer>&);
    void maybeLoadFallbackSynchronously(const ResourceRequest&, ResourceError&, ResourceResponse&, RefPtr<SharedBuffer>&);

    bool canCacheInBackForwardCache();

    Status status() const;
    bool update();
    bool swapCache();
    void abort();

    void setDOMApplicationCache(DOMApplicationCache*);
    void notifyDOMApplicationCache(const AtomString& eventType, int progressTotal, int progressDone);

    void stopLoadingInFrame(LocalFrame&);

    void stopDeferringEvents(); // Also raises the events that have been queued up.

    Vector<ResourceInfo> resourceList();
    CacheInfo applicationCacheInfo();

    bool shouldLoadResourceFromApplicationCache(const ResourceRequest&, RefPtr<ApplicationCacheResource>&);
    bool getApplicationCacheFallbackResource(const ResourceRequest&, RefPtr<ApplicationCacheResource>&, ApplicationCache* = nullptr);

private:
    friend class ApplicationCacheGroup;

    struct DeferredEvent {
        AtomString eventType;
        int progressTotal;
        int progressDone;
    };

    bool isApplicationCacheEnabled();
    bool isApplicationCacheBlockedForRequest(const ResourceRequest&);

    void dispatchDOMEvent(const AtomString& eventType, int progressTotal, int progressDone);

    bool scheduleLoadFallbackResourceFromApplicationCache(ResourceLoader*, ApplicationCache* = nullptr);
    void setCandidateApplicationCacheGroup(ApplicationCacheGroup*);
    ApplicationCacheGroup* candidateApplicationCacheGroup() const;
    void setApplicationCache(RefPtr<ApplicationCache>&&);
    ApplicationCache* applicationCache() const { return m_applicationCache.get(); }
    ApplicationCache* mainResourceApplicationCache() const { return m_mainResourceApplicationCache.get(); }
    bool maybeLoadFallbackForMainError(const ResourceRequest&, const ResourceError&);

    WeakPtr<DOMApplicationCache, WeakPtrImplWithEventTargetData> m_domApplicationCache;
    SingleThreadWeakRef<DocumentLoader> m_documentLoader;

    bool m_defersEvents { true }; // Events are deferred until after document onload.
    Vector<DeferredEvent> m_deferredEvents;

    // The application cache that the document loader is associated with (if any).
    RefPtr<ApplicationCache> m_applicationCache;

    // Before an application cache has finished loading, this will be the candidate application
    // group that the document loader is associated with.
    WeakPtr<ApplicationCacheGroup> m_candidateApplicationCacheGroup;

    // This is the application cache the main resource was loaded from (if any).
    RefPtr<ApplicationCache> m_mainResourceApplicationCache;
};

}  // namespace WebCore
