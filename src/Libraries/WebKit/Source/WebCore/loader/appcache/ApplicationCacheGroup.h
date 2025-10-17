/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 24, 2024.
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

#include "ApplicationCacheResourceLoader.h"
#include "DOMApplicationCache.h"
#include <wtf/Noncopyable.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/URL.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {
class ApplicationCacheGroup;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::ApplicationCacheGroup> : std::true_type { };
}

namespace WebCore {

class ApplicationCache;
class ApplicationCacheResource;
class ApplicationCacheStorage;
class Document;
class DocumentLoader;
class LocalFrame;
class SecurityOrigin;

enum ApplicationCacheUpdateOption {
    ApplicationCacheUpdateWithBrowsingContext,
    ApplicationCacheUpdateWithoutBrowsingContext
};

class ApplicationCacheGroup : public CanMakeWeakPtr<ApplicationCacheGroup> {
    WTF_MAKE_NONCOPYABLE(ApplicationCacheGroup);
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(Loader);
public:
    explicit ApplicationCacheGroup(Ref<ApplicationCacheStorage>&&, const URL& manifestURL);
    virtual ~ApplicationCacheGroup();
    
    enum UpdateStatus { Idle, Checking, Downloading };

    static ApplicationCache* cacheForMainRequest(const ResourceRequest&, DocumentLoader*);
    static ApplicationCache* fallbackCacheForMainRequest(const ResourceRequest&, DocumentLoader*);
    
    static void selectCache(LocalFrame&, const URL& manifestURL);
    static void selectCacheWithoutManifestURL(LocalFrame&);

    ApplicationCacheStorage& storage() { return m_storage; }
    const URL& manifestURL() const { return m_manifestURL; }
    const SecurityOrigin& origin() const { return m_origin.get(); }
    UpdateStatus updateStatus() const { return m_updateStatus; }
    void setUpdateStatus(UpdateStatus status);

    void setStorageID(unsigned storageID) { m_storageID = storageID; }
    unsigned storageID() const { return m_storageID; }
    void clearStorageID();
    
    void update(LocalFrame&, ApplicationCacheUpdateOption); // FIXME: Frame should not be needed when updating without browsing context.
    void cacheDestroyed(ApplicationCache&);
    
    void abort(LocalFrame&);

    bool cacheIsComplete(ApplicationCache& cache) { return m_caches.contains(&cache); }

    void stopLoadingInFrame(LocalFrame&);

    ApplicationCache* newestCache() const { return m_newestCache.get(); }
    void setNewestCache(Ref<ApplicationCache>&&);

    void makeObsolete();
    bool isObsolete() const { return m_isObsolete; }

    void finishedLoadingMainResource(DocumentLoader&);
    void failedLoadingMainResource(DocumentLoader&);

    void disassociateDocumentLoader(DocumentLoader&);

private:
    static void postListenerTask(const AtomString& eventType, const UncheckedKeyHashSet<DocumentLoader*>& set) { postListenerTask(eventType, 0, 0, set); }
    static void postListenerTask(const AtomString& eventType, DocumentLoader& loader)  { postListenerTask(eventType, 0, 0, loader); }
    static void postListenerTask(const AtomString& eventType, int progressTotal, int progressDone, const UncheckedKeyHashSet<DocumentLoader*>&);
    static void postListenerTask(const AtomString& eventType, int progressTotal, int progressDone, DocumentLoader&);

    void scheduleReachedMaxAppCacheSizeCallback();

    void didFinishLoadingManifest();
    void didFailLoadingManifest(ApplicationCacheResourceLoader::Error);

    void didFailLoadingEntry(ApplicationCacheResourceLoader::Error, const URL&, unsigned type);
    void didFinishLoadingEntry(const URL&);

    void didReachMaxAppCacheSize();
    void didReachOriginQuota(int64_t totalSpaceNeeded);
    
    void startLoadingEntry();
    void deliverDelayedMainResources();
    void checkIfLoadIsComplete();
    void cacheUpdateFailed();
    void recalculateAvailableSpaceInQuota();
    void manifestNotFound();
    
    void addEntry(const String&, unsigned type);
    
    void associateDocumentLoaderWithCache(DocumentLoader*, ApplicationCache*);
    
    void stopLoading();

    ResourceRequest createRequest(URL&&, ApplicationCacheResource*);

    Ref<ApplicationCacheStorage> m_storage;

    URL m_manifestURL;
    Ref<SecurityOrigin> m_origin;
    UpdateStatus m_updateStatus { Idle };
    
    // This is the newest complete cache in the group.
    RefPtr<ApplicationCache> m_newestCache;
    
    // All complete caches in this cache group.
    UncheckedKeyHashSet<ApplicationCache*> m_caches;
    
    // The cache being updated (if any). Note that cache updating does not immediately create a new
    // ApplicationCache object, so this may be null even when update status is not Idle.
    RefPtr<ApplicationCache> m_cacheBeingUpdated;

    // List of pending master entries, used during the update process to ensure that new master entries are cached.
    UncheckedKeyHashSet<DocumentLoader*> m_pendingMasterResourceLoaders;
    // How many of the above pending master entries have not yet finished downloading.
    int m_downloadingPendingMasterResourceLoadersCount { 0 };
    
    // These are all the document loaders that are associated with a cache in this group.
    UncheckedKeyHashSet<DocumentLoader*> m_associatedDocumentLoaders;

    // The URLs and types of pending cache entries.
    HashMap<String, unsigned> m_pendingEntries;
    
    // The total number of items to be processed to update the cache group and the number that have been done.
    int m_progressTotal { 0 };
    int m_progressDone { 0 };

    // Frame used for fetching resources when updating.
    // FIXME: An update started by a particular frame should not stop if it is destroyed, but there are other frames associated with the same cache group.
    WeakPtr<LocalFrame> m_frame;
  
    // An obsolete cache group is never stored, but the opposite is not true - storing may fail for multiple reasons, such as exceeding disk quota.
    unsigned m_storageID { 0 };
    bool m_isObsolete { false };

    // During update, this is used to handle asynchronously arriving results.
    enum CompletionType {
        None,
        NoUpdate,
        Failure,
        Completed
    };
    CompletionType m_completionType { None };

    // This flag is set immediately after the ChromeClient::reachedMaxAppCacheSize() callback is invoked as a result of the storage layer failing to save a cache
    // due to reaching the maximum size of the application cache database file. This flag is used by ApplicationCacheGroup::checkIfLoadIsComplete() to decide
    // the course of action in case of this failure (i.e. call the ChromeClient callback or run the failure steps).
    bool m_calledReachedMaxAppCacheSize { false };
    
    RefPtr<ApplicationCacheResource> m_currentResource;
    RefPtr<ApplicationCacheResourceLoader> m_entryLoader;
    Markable<ResourceLoaderIdentifier> m_currentResourceIdentifier;

    RefPtr<ApplicationCacheResource> m_manifestResource;
    RefPtr<ApplicationCacheResourceLoader> m_manifestLoader;

    int64_t m_availableSpaceInQuota;
    bool m_originQuotaExceededPreviously { false };

    friend class ChromeClientCallbackTimer;
};

} // namespace WebCore
