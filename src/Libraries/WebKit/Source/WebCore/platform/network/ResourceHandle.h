/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 8, 2023.
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

#include "AuthenticationClient.h"
#include "ResourceLoaderOptions.h"
#include "StoredCredentialsPolicy.h"
#include <wtf/Box.h>
#include <wtf/MonotonicTime.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/text/AtomString.h>

#if PLATFORM(COCOA)
#include <wtf/RetainPtr.h>
#endif

#if USE(CF)
typedef const struct __CFData * CFDataRef;
#endif

#if PLATFORM(COCOA)
OBJC_CLASS NSCachedURLResponse;
OBJC_CLASS NSData;
OBJC_CLASS NSDictionary;
OBJC_CLASS NSError;
OBJC_CLASS NSURLConnection;
OBJC_CLASS NSURLRequest;
#endif

#if PLATFORM(COCOA)
typedef const struct __CFURLStorageSession* CFURLStorageSessionRef;
#endif

namespace WTF {
class SchedulePair;
template<typename T> class MessageQueue;
}

namespace WebCore {

class AuthenticationChallenge;
class Credential;
class LocalFrame;
class NetworkingContext;
class ProtectionSpace;
class ResourceError;
class ResourceHandleClient;
class ResourceHandleInternal;
class NetworkLoadMetrics;
class ResourceRequest;
class ResourceResponse;
class SecurityOrigin;
class FragmentedSharedBuffer;
class SynchronousLoaderMessageQueue;
class Timer;

class ResourceHandle : public RefCounted<ResourceHandle>, public AuthenticationClient {
public:
    WEBCORE_EXPORT static RefPtr<ResourceHandle> create(NetworkingContext*, const ResourceRequest&, ResourceHandleClient*, bool defersLoading, bool shouldContentSniff, ContentEncodingSniffingPolicy, RefPtr<SecurityOrigin>&& sourceOrigin, bool isMainFrameNavigation);
    WEBCORE_EXPORT static void loadResourceSynchronously(NetworkingContext*, const ResourceRequest&, StoredCredentialsPolicy, SecurityOrigin*, ResourceError&, ResourceResponse&, Vector<uint8_t>& data);
    WEBCORE_EXPORT virtual ~ResourceHandle();

#if PLATFORM(COCOA)
    void willSendRequest(ResourceRequest&&, ResourceResponse&&, CompletionHandler<void(ResourceRequest&&)>&&);
#endif

    void didReceiveResponse(ResourceResponse&&, CompletionHandler<void()>&&);

    bool shouldUseCredentialStorage();
    void didReceiveAuthenticationChallenge(const AuthenticationChallenge&);
    void receivedCredential(const AuthenticationChallenge&, const Credential&) override;
    void receivedRequestToContinueWithoutCredential(const AuthenticationChallenge&) override;
    void receivedCancellation(const AuthenticationChallenge&) override;
    void receivedRequestToPerformDefaultHandling(const AuthenticationChallenge&) override;
    void receivedChallengeRejection(const AuthenticationChallenge&) override;

#if PLATFORM(COCOA)
    bool tryHandlePasswordBasedAuthentication(const AuthenticationChallenge&);
#endif

#if PLATFORM(COCOA) && USE(PROTECTION_SPACE_AUTH_CALLBACK)
    void canAuthenticateAgainstProtectionSpace(const ProtectionSpace&, CompletionHandler<void(bool)>&&);
#endif

#if PLATFORM(COCOA)
    WEBCORE_EXPORT NSURLConnection *connection() const;
    id makeDelegate(bool, RefPtr<SynchronousLoaderMessageQueue>&&);
    id delegate();
    void releaseDelegate();
#endif

#if PLATFORM(COCOA)
    void schedule(WTF::SchedulePair&);
    void unschedule(WTF::SchedulePair&);
#endif

    bool shouldContentSniff() const;
    static bool shouldContentSniffURL(const URL&);

    ContentEncodingSniffingPolicy contentEncodingSniffingPolicy() const;

    WEBCORE_EXPORT static void forceContentSniffing();

    bool hasAuthenticationChallenge() const;
    void clearAuthentication();
    WEBCORE_EXPORT virtual void cancel();

    NetworkLoadMetrics* networkLoadMetrics();
    void setNetworkLoadMetrics(Box<NetworkLoadMetrics>&&);

    MonotonicTime startTimeBeforeRedirects() const;
    bool failsTAOCheck() const;
    void checkTAO(const ResourceResponse&);
    bool hasCrossOriginRedirect() const;
    void markAsHavingCrossOriginRedirect();
    uint16_t redirectCount() const;
    void incrementRedirectCount();

    // The client may be 0, in which case no callbacks will be made.
    WEBCORE_EXPORT ResourceHandleClient* client() const;
    WEBCORE_EXPORT void clearClient();

    WEBCORE_EXPORT void setDefersLoading(bool);

    WEBCORE_EXPORT ResourceRequest& firstRequest();
    const String& lastHTTPMethod() const;

    void failureTimerFired();

    NetworkingContext* context() const;

    using RefCounted<ResourceHandle>::ref;
    using RefCounted<ResourceHandle>::deref;

#if PLATFORM(COCOA)
    WEBCORE_EXPORT static CFStringRef synchronousLoadRunLoopMode();
#endif

    typedef Ref<ResourceHandle> (*BuiltinConstructor)(const ResourceRequest& request, ResourceHandleClient* client);
    static void registerBuiltinConstructor(const AtomString& protocol, BuiltinConstructor);

    typedef void (*BuiltinSynchronousLoader)(NetworkingContext*, const ResourceRequest&, StoredCredentialsPolicy, ResourceError&, ResourceResponse&, Vector<uint8_t>& data);
    static void registerBuiltinSynchronousLoader(const AtomString& protocol, BuiltinSynchronousLoader);

protected:
    ResourceHandle(NetworkingContext*, const ResourceRequest&, ResourceHandleClient*, bool defersLoading, bool shouldContentSniff, ContentEncodingSniffingPolicy, RefPtr<SecurityOrigin>&& sourceOrigin, bool isMainFrameNavigation);

private:
    enum FailureType {
        NoFailure,
        BlockedFailure,
        InvalidURLFailure
    };

    void platformSetDefersLoading(bool);

    void scheduleFailure(FailureType);

    bool start();
    static void platformLoadResourceSynchronously(NetworkingContext*, const ResourceRequest&, StoredCredentialsPolicy, SecurityOrigin*, ResourceError&, ResourceResponse&, Vector<uint8_t>& data);

    void refAuthenticationClient() override { ref(); }
    void derefAuthenticationClient() override { deref(); }

#if PLATFORM(COCOA)
    enum class SchedulingBehavior { Asynchronous, Synchronous };
#endif

#if PLATFORM(MAC)
    void createNSURLConnection(id delegate, bool shouldUseCredentialStorage, bool shouldContentSniff, ContentEncodingSniffingPolicy, SchedulingBehavior);
#endif

#if PLATFORM(IOS_FAMILY)
    void createNSURLConnection(id delegate, bool shouldUseCredentialStorage, bool shouldContentSniff, ContentEncodingSniffingPolicy, SchedulingBehavior, NSDictionary *connectionProperties);
#endif

#if PLATFORM(COCOA)
    NSURLRequest *applySniffingPoliciesIfNeeded(NSURLRequest *, bool shouldContentSniff, ContentEncodingSniffingPolicy);
#endif

    friend class ResourceHandleInternal;
    std::unique_ptr<ResourceHandleInternal> d;
};

}
