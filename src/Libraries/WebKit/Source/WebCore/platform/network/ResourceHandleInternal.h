/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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

#include "AuthenticationChallenge.h"
#include "NetworkingContext.h"
#include "ResourceHandle.h"
#include "ResourceHandleClient.h"
#include "ResourceRequest.h"
#include "Timer.h"
#include <wtf/MonotonicTime.h>

#if PLATFORM(COCOA)
OBJC_CLASS NSURLAuthenticationChallenge;
OBJC_CLASS NSURLConnection;

typedef const struct __CFURLStorageSession* CFURLStorageSessionRef;
#endif

// The allocations and releases in ResourceHandleInternal are
// Cocoa-exception-free (either simple Foundation classes or
// WebCoreResourceLoaderImp which avoids doing work in dealloc).

namespace WebCore {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(ResourceHandleInternal);
class ResourceHandleInternal {
    WTF_MAKE_NONCOPYABLE(ResourceHandleInternal);
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(ResourceHandleInternal);
public:
    ResourceHandleInternal(ResourceHandle* loader, NetworkingContext* context, const ResourceRequest& request, ResourceHandleClient* client, bool defersLoading, bool shouldContentSniff, ContentEncodingSniffingPolicy contentEncodingSniffingPolicy, RefPtr<SecurityOrigin>&& sourceOrigin, bool isMainFrameNavigation)
        : m_context(context)
        , m_client(client)
        , m_firstRequest(request)
        , m_lastHTTPMethod(request.httpMethod())
        , m_partition(request.cachePartition())
        , m_failureTimer(*loader, &ResourceHandle::failureTimerFired)
        , m_sourceOrigin(WTFMove(sourceOrigin))
        , m_contentEncodingSniffingPolicy(contentEncodingSniffingPolicy)
        , m_defersLoading(defersLoading)
        , m_shouldContentSniff(shouldContentSniff)
        , m_isMainFrameNavigation(isMainFrameNavigation)
    {
        const URL& url = m_firstRequest.url();
        m_user = url.user();
        m_password = url.password();
        m_firstRequest.removeCredentials();
    }
    
    ~ResourceHandleInternal();

    ResourceHandleClient* client() { return m_client; }

    RefPtr<NetworkingContext> m_context;
    ResourceHandleClient* m_client;
    ResourceRequest m_firstRequest;
    ResourceRequest m_previousRequest;
    String m_lastHTTPMethod;
    String m_partition;

    // Suggested credentials for the current redirection step.
    String m_user;
    String m_password;
    
    Credential m_initialCredential;
    
#if PLATFORM(COCOA)
    RetainPtr<NSURLConnection> m_connection;
    RetainPtr<id> m_delegate;
    RetainPtr<CFURLStorageSessionRef> m_storageSession;

    // We need to keep a reference to the original challenge to be able to cancel it.
    // It is almost identical to m_currentWebChallenge.nsURLAuthenticationChallenge(), but has a different sender.
    NSURLAuthenticationChallenge *m_currentMacChallenge { nil };
#endif
    Box<NetworkLoadMetrics> m_networkLoadMetrics;
    MonotonicTime m_startTime;

    AuthenticationChallenge m_currentWebChallenge;
    Timer m_failureTimer;
    RefPtr<SecurityOrigin> m_sourceOrigin;

    int status { 0 };

    uint16_t m_redirectCount { 0 };

    ResourceHandle::FailureType m_scheduledFailureType { ResourceHandle::NoFailure };
    ContentEncodingSniffingPolicy m_contentEncodingSniffingPolicy;

    bool m_defersLoading;
    bool m_shouldContentSniff;
    bool m_failsTAOCheck { false };
    bool m_hasCrossOriginRedirect { false };
    bool m_isCrossOrigin { false };
    bool m_isMainFrameNavigation { false };
#if PLATFORM(COCOA)
    bool m_startWhenScheduled { false };
#endif
};

} // namespace WebCore
