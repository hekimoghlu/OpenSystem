/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 16, 2024.
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

#include "PolicyChecker.h"
#include "SharedBuffer.h"
#include <wtf/CompletionHandler.h>
#include <wtf/Expected.h>
#include <wtf/Lock.h>
#include <wtf/MainThreadDispatcher.h>
#include <wtf/Noncopyable.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/ThreadSafeRefCounted.h>

namespace WebCore {

class PlatformMediaResource;
class ResourceError;
class ResourceRequest;
class ResourceResponse;

class PlatformMediaResourceClient : public ThreadSafeRefCounted<PlatformMediaResourceClient> {
public:
    virtual ~PlatformMediaResourceClient() = default;

    // Those methods must be called on PlatformMediaResourceLoader::targetDispatcher()
    virtual void responseReceived(PlatformMediaResource&, const ResourceResponse&, CompletionHandler<void(ShouldContinuePolicyCheck)>&& completionHandler) { completionHandler(ShouldContinuePolicyCheck::Yes); }
    virtual void redirectReceived(PlatformMediaResource&, ResourceRequest&& request, const ResourceResponse&, CompletionHandler<void(ResourceRequest&&)>&& completionHandler) { completionHandler(WTFMove(request)); }
    virtual bool shouldCacheResponse(PlatformMediaResource&, const ResourceResponse&) { return true; }
    virtual void dataSent(PlatformMediaResource&, unsigned long long, unsigned long long) { }
    virtual void dataReceived(PlatformMediaResource&, const SharedBuffer&) { RELEASE_ASSERT_NOT_REACHED(); }
    virtual void accessControlCheckFailed(PlatformMediaResource&, const ResourceError&) { }
    virtual void loadFailed(PlatformMediaResource&, const ResourceError&) { }
    virtual void loadFinished(PlatformMediaResource&, const NetworkLoadMetrics&) { }
};

class PlatformMediaResourceLoader : public ThreadSafeRefCounted<PlatformMediaResourceLoader, WTF::DestructionThread::Main> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(PlatformMediaResourceLoader);
    WTF_MAKE_NONCOPYABLE(PlatformMediaResourceLoader);
public:
    enum LoadOption {
        BufferData = 1 << 0,
        DisallowCaching = 1 << 1,
    };
    typedef unsigned LoadOptions;

    virtual ~PlatformMediaResourceLoader() = default;

    virtual void sendH2Ping(const URL&, CompletionHandler<void(Expected<Seconds, ResourceError>&&)>&&) = 0;

    // Can be called on any threads. Return the function dispatcher on which the PlaftormMediaResource and PlatformMediaResourceClient must be be called on.
    virtual Ref<GuaranteedSerialFunctionDispatcher> targetDispatcher() { return MainThreadDispatcher::singleton(); }
    // requestResource will be called on the main thread, the PlatformMediaResource object is to be used on targetDispatcher().
    virtual RefPtr<PlatformMediaResource> requestResource(ResourceRequest&&, LoadOptions) = 0;

protected:
    PlatformMediaResourceLoader() = default;
};

class PlatformMediaResource : public ThreadSafeRefCountedAndCanMakeThreadSafeWeakPtr<PlatformMediaResource> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(PlatformMediaResource);
    WTF_MAKE_NONCOPYABLE(PlatformMediaResource);
public:
    // Called on the main thread.
    PlatformMediaResource() = default;

    // Can be called on any threads, must be made thread-safe.
    virtual bool didPassAccessControlCheck() const { return false; }

    // Can be called on any thread.
    virtual ~PlatformMediaResource() = default;
    virtual void shutdown() { }
    void setClient(RefPtr<PlatformMediaResourceClient>&& client)
    {
        Locker locker { m_lock };
        m_client = WTFMove(client);
    }
    RefPtr<PlatformMediaResourceClient> client() const
    {
        Locker locker { m_lock };
        return m_client;
    }

private:
    RefPtr<PlatformMediaResourceClient> m_client WTF_GUARDED_BY_LOCK(m_lock);
    mutable Lock m_lock;
};

} // namespace WebCore

#endif
