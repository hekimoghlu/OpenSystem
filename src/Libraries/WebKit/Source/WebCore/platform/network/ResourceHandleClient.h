/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 6, 2024.
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

#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/Ref.h>

#if PLATFORM(IOS_FAMILY)
#include <wtf/RetainPtr.h>
#endif

#if PLATFORM(COCOA)
OBJC_CLASS NSCachedURLResponse;
#endif

namespace WebCore {
class AuthenticationChallenge;
class Credential;
class FragmentedSharedBuffer;
class NetworkLoadMetrics;
class ProtectionSpace;
class ResourceHandle;
class ResourceError;
class ResourceRequest;
class ResourceResponse;
class SharedBuffer;

enum CacheStoragePolicy {
    StorageAllowed,
    StorageAllowedInMemoryOnly,
    StorageNotAllowed
};

class ResourceHandleClient {
public:
    WEBCORE_EXPORT ResourceHandleClient();
    WEBCORE_EXPORT virtual ~ResourceHandleClient();

    virtual void didSendData(ResourceHandle*, unsigned long long /*bytesSent*/, unsigned long long /*totalBytesToBeSent*/) { }

    virtual void didReceiveData(ResourceHandle*, const SharedBuffer&, int /*encodedDataLength*/) { }
    WEBCORE_EXPORT virtual void didReceiveBuffer(ResourceHandle*, const FragmentedSharedBuffer&, int encodedDataLength);

    virtual void didFinishLoading(ResourceHandle*, const NetworkLoadMetrics&) { }
    virtual void didFail(ResourceHandle*, const ResourceError&) { }
    virtual void wasBlocked(ResourceHandle*) { }
    virtual void cannotShowURL(ResourceHandle*) { }

    virtual bool loadingSynchronousXHR() { return false; }

    WEBCORE_EXPORT virtual void willSendRequestAsync(ResourceHandle*, ResourceRequest&&, ResourceResponse&&, CompletionHandler<void(ResourceRequest&&)>&&) = 0;

    WEBCORE_EXPORT virtual void didReceiveResponseAsync(ResourceHandle*, ResourceResponse&&, CompletionHandler<void()>&&) = 0;

#if USE(PROTECTION_SPACE_AUTH_CALLBACK)
    WEBCORE_EXPORT virtual void canAuthenticateAgainstProtectionSpaceAsync(ResourceHandle*, const ProtectionSpace&, CompletionHandler<void(bool)>&&) = 0;
#endif

#if PLATFORM(COCOA)
    virtual void willCacheResponseAsync(ResourceHandle*, NSCachedURLResponse *response, CompletionHandler<void(NSCachedURLResponse *)>&& completionHandler) { completionHandler(response); }
#endif

    virtual bool shouldUseCredentialStorage(ResourceHandle*) { return false; }
    virtual void didReceiveAuthenticationChallenge(ResourceHandle*, const AuthenticationChallenge&) { }
    virtual void receivedCancellation(ResourceHandle*, const AuthenticationChallenge&) { }

#if PLATFORM(IOS_FAMILY)
    virtual RetainPtr<CFDictionaryRef> connectionProperties(ResourceHandle*) { return nullptr; }
#endif
};

}
