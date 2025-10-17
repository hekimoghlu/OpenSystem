/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 24, 2024.
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

#if ENABLE(VIDEO)

#include "CachedRawResourceClient.h"
#include "CachedResourceHandle.h"
#include "ContextDestructionObserver.h"
#include "FetchOptions.h"
#include "PlatformMediaResourceLoader.h"
#include "ResourceResponse.h"
#include <wtf/Atomics.h>
#include <wtf/HashSet.h>
#include <wtf/Ref.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class CachedRawResource;
class Document;
class Element;
class MediaResource;
class WeakPtrImplWithEventTargetData;

class MediaResourceLoader final : public PlatformMediaResourceLoader, public CanMakeWeakPtr<MediaResourceLoader>, public ContextDestructionObserver {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(MediaResourceLoader, WEBCORE_EXPORT);
public:
    static Ref<MediaResourceLoader> create(Document& document, Element& element, const String& crossOriginMode, FetchOptions::Destination destination) { return adoptRef(*new MediaResourceLoader(document, element, crossOriginMode, destination)); }
    WEBCORE_EXPORT virtual ~MediaResourceLoader();

    RefPtr<PlatformMediaResource> requestResource(ResourceRequest&&, LoadOptions) final;
    void sendH2Ping(const URL&, CompletionHandler<void(Expected<Seconds, ResourceError>&&)>&&) final;
    void removeResource(MediaResource&);

    Document* document();
    RefPtr<Document> protectedDocument();
    const String& crossOriginMode() const;

    WEBCORE_EXPORT static void recordResponsesForTesting();
    WEBCORE_EXPORT Vector<ResourceResponse> responsesForTesting() const;
    void addResponseForTesting(const ResourceResponse&);

    bool verifyMediaResponse(const URL& requestURL, const ResourceResponse&, const SecurityOrigin*);

private:
    WEBCORE_EXPORT MediaResourceLoader(Document&, Element&, const String& crossOriginMode, FetchOptions::Destination);

    void contextDestroyed() final;

    WeakPtr<Document, WeakPtrImplWithEventTargetData> m_document WTF_GUARDED_BY_CAPABILITY(mainThread);
    WeakPtr<Element, WeakPtrImplWithEventTargetData> m_element WTF_GUARDED_BY_CAPABILITY(mainThread);
    String m_crossOriginMode WTF_GUARDED_BY_CAPABILITY(mainThread);
    SingleThreadWeakHashSet<MediaResource> m_resources WTF_GUARDED_BY_CAPABILITY(mainThread);
    Vector<ResourceResponse> m_responsesForTesting WTF_GUARDED_BY_CAPABILITY(mainThread);
    FetchOptions::Destination m_destination WTF_GUARDED_BY_CAPABILITY(mainThread);

    struct ValidationInformation {
        RefPtr<const SecurityOrigin> origin;
        bool usedOpaqueResponse { false };
        bool usedServiceWorker { false };
    };
    HashMap<URL, ValidationInformation> m_validationLoadInformations WTF_GUARDED_BY_CAPABILITY(mainThread);
};

class MediaResource : public PlatformMediaResource, public CachedRawResourceClient {
    WTF_MAKE_TZONE_ALLOCATED(MediaResource);
public:
    static Ref<MediaResource> create(MediaResourceLoader&, CachedResourceHandle<CachedRawResource>&&);
    virtual ~MediaResource();

    // PlatformMediaResource
    void shutdown() override;
    bool didPassAccessControlCheck() const override { return m_didPassAccessControlCheck.load(); }

    // CachedRawResourceClient
    void responseReceived(CachedResource&, const ResourceResponse&, CompletionHandler<void()>&&) override;
    void redirectReceived(CachedResource&, ResourceRequest&&, const ResourceResponse&, CompletionHandler<void(ResourceRequest&&)>&&) override;
    bool shouldCacheResponse(CachedResource&, const ResourceResponse&) override;
    void dataSent(CachedResource&, unsigned long long, unsigned long long) override;
    void dataReceived(CachedResource&, const SharedBuffer&) override;
    void notifyFinished(CachedResource&, const NetworkLoadMetrics&, LoadWillContinueInAnotherProcess) override;

private:
    Ref<MediaResourceLoader> protectedLoader() const;
    CachedResourceHandle<CachedRawResource> protectedResource() const;

    MediaResource(MediaResourceLoader&, CachedResourceHandle<CachedRawResource>&&);
    void ensureShutdown();
    const Ref<MediaResourceLoader> m_loader;
    Atomic<bool> m_didPassAccessControlCheck { false };
    CachedResourceHandle<CachedRawResource> m_resource;
};

} // namespace WebCore

#endif // ENABLE(VIDEO)
