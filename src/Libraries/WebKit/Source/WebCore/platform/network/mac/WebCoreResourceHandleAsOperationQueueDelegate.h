/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 31, 2023.
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
#import <dispatch/dispatch.h>
#import <wtf/Box.h>
#import <wtf/Function.h>
#import <wtf/Lock.h>
#import <wtf/MessageQueue.h>
#import <wtf/RefPtr.h>
#import <wtf/RetainPtr.h>
#import <wtf/SchedulePair.h>
#import <wtf/threads/BinarySemaphore.h>

namespace WebCore {
class NetworkLoadMetrics;
class ResourceHandle;
class SynchronousLoaderMessageQueue;
}

@interface WebCoreResourceHandleAsOperationQueueDelegate : NSObject <NSURLConnectionDelegate> {
    Lock m_lock;
    WebCore::ResourceHandle* m_handle WTF_GUARDED_BY_LOCK(m_lock);

    // Synchronous delegates on operation queue wait until main thread sends an asynchronous response.
    BinarySemaphore m_semaphore;
    RefPtr<WebCore::SynchronousLoaderMessageQueue> m_messageQueue;
    RetainPtr<NSURLRequest> m_requestResult;
    RetainPtr<NSCachedURLResponse> m_cachedResponseResult;
    std::optional<SchedulePairHashSet> m_scheduledPairs;
    BOOL m_boolResult;
}

- (void)detachHandle;
- (id)initWithHandle:(WebCore::ResourceHandle*)handle messageQueue:(RefPtr<WebCore::SynchronousLoaderMessageQueue>&&)messageQueue;
@end

@interface WebCoreResourceHandleWithCredentialStorageAsOperationQueueDelegate : WebCoreResourceHandleAsOperationQueueDelegate

@end
