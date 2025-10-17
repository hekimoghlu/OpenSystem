/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 23, 2022.
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

#include <WebCore/SharedWorkerIdentifier.h>
#include <WebCore/SharedWorkerKey.h>
#include <WebCore/SharedWorkerObjectIdentifier.h>
#include <WebCore/TransferredMessagePort.h>
#include <WebCore/WorkerFetchResult.h>
#include <WebCore/WorkerInitializationData.h>
#include <WebCore/WorkerOptions.h>
#include <wtf/CheckedRef.h>
#include <wtf/Identified.h>
#include <wtf/ListHashSet.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakPtr.h>

namespace WebKit {
class WebSharedWorker;
}

namespace WebCore {
class RegistrableDomain;
class Site;
}

namespace WebKit {

class WebSharedWorkerServer;
class WebSharedWorkerServerToContextConnection;

class WebSharedWorker : public RefCountedAndCanMakeWeakPtr<WebSharedWorker>, public Identified<WebCore::SharedWorkerIdentifier> {
    WTF_MAKE_TZONE_ALLOCATED(WebSharedWorker);
public:
    static Ref<WebSharedWorker> create(WebSharedWorkerServer&, const WebCore::SharedWorkerKey&, const WebCore::WorkerOptions&);

    ~WebSharedWorker();

    static WebSharedWorker* fromIdentifier(WebCore::SharedWorkerIdentifier);

    const WebCore::SharedWorkerKey& key() const { return m_key; }
    const WebCore::WorkerOptions& workerOptions() const { return m_workerOptions; }
    const WebCore::ClientOrigin& origin() const { return m_key.origin; }
    const URL& url() const { return m_key.url; }
    WebCore::RegistrableDomain topRegistrableDomain() const;
    WebCore::Site topSite() const;
    WebSharedWorkerServerToContextConnection* contextConnection() const;

    void addSharedWorkerObject(WebCore::SharedWorkerObjectIdentifier, const WebCore::TransferredMessagePort&);
    void removeSharedWorkerObject(WebCore::SharedWorkerObjectIdentifier);
    void suspend(WebCore::SharedWorkerObjectIdentifier);
    void resume(WebCore::SharedWorkerObjectIdentifier);
    unsigned sharedWorkerObjectsCount() const { return m_sharedWorkerObjects.size(); }
    void forEachSharedWorkerObject(const Function<void(WebCore::SharedWorkerObjectIdentifier, const WebCore::TransferredMessagePort&)>&) const;
    std::optional<WebCore::ProcessIdentifier> firstSharedWorkerObjectProcess() const;

    void didCreateContextConnection(WebSharedWorkerServerToContextConnection&);

    bool isRunning() const { return m_isRunning; }
    void markAsRunning() { m_isRunning = true; }

    const WebCore::WorkerInitializationData& initializationData() const { return m_initializationData; }
    void setInitializationData(WebCore::WorkerInitializationData&& initializationData) { m_initializationData = WTFMove(initializationData); }

    const WebCore::WorkerFetchResult& fetchResult() const { return m_fetchResult; }
    void setFetchResult(WebCore::WorkerFetchResult&&);
    bool didFinishFetching() const { return !!m_fetchResult.script; }

    void launch(WebSharedWorkerServerToContextConnection&);

    struct SharedWorkerObjectState {
        bool isSuspended { false };
        std::optional<WebCore::TransferredMessagePort> port;
    };

    struct Object {
        WebCore::SharedWorkerObjectIdentifier identifier;
        SharedWorkerObjectState state;
    };

private:
    WebSharedWorker(WebSharedWorkerServer&, const WebCore::SharedWorkerKey&, const WebCore::WorkerOptions&);

    WebSharedWorker(WebSharedWorker&&) = delete;
    WebSharedWorker& operator=(WebSharedWorker&&) = delete;
    WebSharedWorker(const WebSharedWorker&) = delete;
    WebSharedWorker& operator=(const WebSharedWorker&) = delete;

    void suspendIfNeeded();
    void resumeIfNeeded();

    WeakPtr<WebSharedWorkerServer> m_server;
    WebCore::SharedWorkerKey m_key;
    WebCore::WorkerOptions m_workerOptions;
    ListHashSet<Object> m_sharedWorkerObjects;
    WebCore::WorkerFetchResult m_fetchResult;
    WebCore::WorkerInitializationData m_initializationData;
    bool m_isRunning { false };
    bool m_isSuspended { false };
};

} // namespace WebKit

namespace WTF {

struct WebSharedWorkerObjectHash {
    static unsigned hash(const WebKit::WebSharedWorker::Object& object) { return DefaultHash<WebCore::SharedWorkerObjectIdentifier>::hash(object.identifier); }
    static bool equal(const WebKit::WebSharedWorker::Object& a, const WebKit::WebSharedWorker::Object& b) { return a.identifier == b.identifier; }
    static constexpr bool safeToCompareToEmptyOrDeleted = true;
};

template<> struct HashTraits<WebKit::WebSharedWorker::Object> : SimpleClassHashTraits<WebSharedWorkerObjectHash> { };
template<> struct DefaultHash<WebKit::WebSharedWorker::Object> : WebSharedWorkerObjectHash { };

} // namespace WTF
