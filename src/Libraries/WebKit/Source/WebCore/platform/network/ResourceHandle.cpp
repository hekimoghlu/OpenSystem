/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 22, 2025.
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
#include "config.h"
#include "ResourceHandle.h"
#include "ResourceHandleInternal.h"

#include "DNS.h"
#include "Logging.h"
#include "NetworkingContext.h"
#include "NotImplemented.h"
#include "ResourceHandleClient.h"
#include "SecurityOrigin.h"
#include "Timer.h"
#include "TimingAllowOrigin.h"
#include <algorithm>
#include <wtf/CompletionHandler.h>
#include <wtf/MainThread.h>
#include <wtf/NeverDestroyed.h>
#include <wtf/text/AtomStringHash.h>
#include <wtf/text/CString.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

static bool shouldForceContentSniffing;

DEFINE_ALLOCATOR_WITH_HEAP_IDENTIFIER(ResourceHandleInternal);

typedef HashMap<AtomString, ResourceHandle::BuiltinConstructor> BuiltinResourceHandleConstructorMap;
static BuiltinResourceHandleConstructorMap& builtinResourceHandleConstructorMap()
{
#if PLATFORM(IOS_FAMILY)
    ASSERT(WebThreadIsLockedOrDisabled());
#else
    ASSERT(isMainThread());
#endif
    static NeverDestroyed<BuiltinResourceHandleConstructorMap> map;
    return map;
}

void ResourceHandle::registerBuiltinConstructor(const AtomString& protocol, ResourceHandle::BuiltinConstructor constructor)
{
    builtinResourceHandleConstructorMap().add(protocol, constructor);
}

typedef HashMap<AtomString, ResourceHandle::BuiltinSynchronousLoader> BuiltinResourceHandleSynchronousLoaderMap;
static BuiltinResourceHandleSynchronousLoaderMap& builtinResourceHandleSynchronousLoaderMap()
{
    ASSERT(isMainThread());
    static NeverDestroyed<BuiltinResourceHandleSynchronousLoaderMap> map;
    return map;
}

void ResourceHandle::registerBuiltinSynchronousLoader(const AtomString& protocol, ResourceHandle::BuiltinSynchronousLoader loader)
{
    builtinResourceHandleSynchronousLoaderMap().add(protocol, loader);
}

ResourceHandle::ResourceHandle(NetworkingContext* context, const ResourceRequest& request, ResourceHandleClient* client, bool defersLoading, bool shouldContentSniff, ContentEncodingSniffingPolicy contentEncodingSniffingPolicy, RefPtr<SecurityOrigin>&& sourceOrigin, bool isMainFrameNavigation)
    : d(makeUnique<ResourceHandleInternal>(this, context, request, client, defersLoading, shouldContentSniff && shouldContentSniffURL(request.url()), contentEncodingSniffingPolicy, WTFMove(sourceOrigin), isMainFrameNavigation))
{
    if (!request.url().isValid()) {
        scheduleFailure(InvalidURLFailure);
        return;
    }

    if (!portAllowed(request.url()) || isIPAddressDisallowed(request.url())) {
        scheduleFailure(BlockedFailure);
        return;
    }
}

RefPtr<ResourceHandle> ResourceHandle::create(NetworkingContext* context, const ResourceRequest& request, ResourceHandleClient* client, bool defersLoading, bool shouldContentSniff, ContentEncodingSniffingPolicy contentEncodingSniffingPolicy, RefPtr<SecurityOrigin>&& sourceOrigin, bool isMainFrameNavigation)
{
    if (auto protocol = request.url().protocol().toExistingAtomString(); !protocol.isNull()) {
        if (auto constructor = builtinResourceHandleConstructorMap().get(protocol))
            return constructor(request, client);
    }

    auto newHandle = adoptRef(*new ResourceHandle(context, request, client, defersLoading, shouldContentSniff, contentEncodingSniffingPolicy, WTFMove(sourceOrigin), isMainFrameNavigation));

    if (newHandle->d->m_scheduledFailureType != NoFailure)
        return newHandle;

    if (newHandle->start())
        return newHandle;

    return nullptr;
}

void ResourceHandle::scheduleFailure(FailureType type)
{
    d->m_scheduledFailureType = type;
    d->m_failureTimer.startOneShot(0_s);
}

void ResourceHandle::failureTimerFired()
{
    if (!client())
        return;

    switch (d->m_scheduledFailureType) {
        case NoFailure:
            ASSERT_NOT_REACHED();
            return;
        case BlockedFailure:
            d->m_scheduledFailureType = NoFailure;
            client()->wasBlocked(this);
            return;
        case InvalidURLFailure:
            d->m_scheduledFailureType = NoFailure;
            client()->cannotShowURL(this);
            return;
    }

    ASSERT_NOT_REACHED();
}

void ResourceHandle::loadResourceSynchronously(NetworkingContext* context, const ResourceRequest& request, StoredCredentialsPolicy storedCredentialsPolicy, SecurityOrigin* sourceOrigin, ResourceError& error, ResourceResponse& response, Vector<uint8_t>& data)
{
    if (auto protocol = request.url().protocol().toExistingAtomString(); !protocol.isNull()) {
        if (auto constructor = builtinResourceHandleSynchronousLoaderMap().get(protocol)) {
            constructor(context, request, storedCredentialsPolicy, error, response, data);
            return;
        }
    }

    platformLoadResourceSynchronously(context, request, storedCredentialsPolicy, sourceOrigin, error, response, data);
}

ResourceHandleClient* ResourceHandle::client() const
{
    return d->m_client;
}

void ResourceHandle::clearClient()
{
    d->m_client = nullptr;
}

void ResourceHandle::didReceiveResponse(ResourceResponse&& response, CompletionHandler<void()>&& completionHandler)
{
    if (response.isHTTP09()) {
        auto url = response.url();
        std::optional<uint16_t> port = url.port();
        if (port && !WTF::isDefaultPortForProtocol(port.value(), url.protocol())) {
            cancel();
            auto message = makeString("Cancelled load from '"_s, url.stringCenterEllipsizedToLength(), "' because it is using HTTP/0.9."_s);
            d->m_client->didFail(this, { String(), 0, url, message });
            completionHandler();
            return;
        }
    }
    client()->didReceiveResponseAsync(this, WTFMove(response), WTFMove(completionHandler));
}

ResourceRequest& ResourceHandle::firstRequest()
{
    return d->m_firstRequest;
}

NetworkingContext* ResourceHandle::context() const
{
    return d->m_context.get();
}

const String& ResourceHandle::lastHTTPMethod() const
{
    return d->m_lastHTTPMethod;
}

bool ResourceHandle::hasAuthenticationChallenge() const
{
    return !d->m_currentWebChallenge.isNull();
}

void ResourceHandle::clearAuthentication()
{
#if PLATFORM(COCOA)
    d->m_currentMacChallenge = nil;
#endif
    d->m_currentWebChallenge.nullify();
}

bool ResourceHandle::failsTAOCheck() const
{
    return d->m_failsTAOCheck;
}

void ResourceHandle::checkTAO(const ResourceResponse& response)
{
    if (d->m_failsTAOCheck)
        return;

    RefPtr<SecurityOrigin> origin;
    if (d->m_isMainFrameNavigation)
        origin = SecurityOrigin::create(firstRequest().url());
    else
        origin = d->m_sourceOrigin;

    if (origin)
        d->m_failsTAOCheck = !passesTimingAllowOriginCheck(response, *origin);
}

bool ResourceHandle::hasCrossOriginRedirect() const
{
    return d->m_hasCrossOriginRedirect;
}

void ResourceHandle::markAsHavingCrossOriginRedirect()
{
    d->m_hasCrossOriginRedirect = true;
}

void ResourceHandle::incrementRedirectCount()
{
    d->m_redirectCount++;
}

uint16_t ResourceHandle::redirectCount() const
{
    return d->m_redirectCount;
}

MonotonicTime ResourceHandle::startTimeBeforeRedirects() const
{
    return d->m_startTime;
}

NetworkLoadMetrics* ResourceHandle::networkLoadMetrics()
{
    return d->m_networkLoadMetrics.get();
}

void ResourceHandle::setNetworkLoadMetrics(Box<NetworkLoadMetrics>&& metrics)
{
    d->m_networkLoadMetrics = WTFMove(metrics);
}

bool ResourceHandle::shouldContentSniff() const
{
    return d->m_shouldContentSniff;
}

ContentEncodingSniffingPolicy ResourceHandle::contentEncodingSniffingPolicy() const
{
    return d->m_contentEncodingSniffingPolicy;
}

bool ResourceHandle::shouldContentSniffURL(const URL& url)
{
#if PLATFORM(COCOA)
    if (shouldForceContentSniffing)
        return true;
#endif
    // We shouldn't content sniff file URLs as their MIME type should be established via their extension.
    return !url.protocolIsFile();
}

void ResourceHandle::forceContentSniffing()
{
    shouldForceContentSniffing = true;
}

void ResourceHandle::setDefersLoading(bool defers)
{
    LOG(Network, "Handle %p setDefersLoading(%s)", this, defers ? "true" : "false");

    ASSERT(d->m_defersLoading != defers); // Deferring is not counted, so calling setDefersLoading() repeatedly is likely to be in error.
    d->m_defersLoading = defers;

    if (defers) {
        ASSERT(d->m_failureTimer.isActive() == (d->m_scheduledFailureType != NoFailure));
        if (d->m_failureTimer.isActive())
            d->m_failureTimer.stop();
    } else if (d->m_scheduledFailureType != NoFailure) {
        ASSERT(!d->m_failureTimer.isActive());
        d->m_failureTimer.startOneShot(0_s);
    }

    platformSetDefersLoading(defers);
}

#if USE(SOUP) || USE(CURL)
ResourceHandleInternal::~ResourceHandleInternal() = default;

ResourceHandle::~ResourceHandle()
{
    ASSERT_NOT_REACHED();
}

bool ResourceHandle::start()
{
    ASSERT_NOT_REACHED();
    return false;
}

void ResourceHandle::cancel()
{
    ASSERT_NOT_REACHED();
}

void ResourceHandle::platformSetDefersLoading(bool)
{
    ASSERT_NOT_REACHED();
}

void ResourceHandle::platformLoadResourceSynchronously(NetworkingContext*, const ResourceRequest&, StoredCredentialsPolicy, SecurityOrigin*, ResourceError&, ResourceResponse&, Vector<uint8_t>&)
{
    ASSERT_NOT_REACHED();
}

bool ResourceHandle::shouldUseCredentialStorage()
{
    ASSERT_NOT_REACHED();
    return false;
}

void ResourceHandle::didReceiveAuthenticationChallenge(const AuthenticationChallenge&)
{
    ASSERT_NOT_REACHED();
}

void ResourceHandle::receivedCredential(const AuthenticationChallenge&, const Credential&)
{
    ASSERT_NOT_REACHED();
}

void ResourceHandle::receivedRequestToContinueWithoutCredential(const AuthenticationChallenge&)
{
    ASSERT_NOT_REACHED();
}

void ResourceHandle::receivedCancellation(const AuthenticationChallenge&)
{
    ASSERT_NOT_REACHED();
}

void ResourceHandle::receivedRequestToPerformDefaultHandling(const AuthenticationChallenge&)
{
    ASSERT_NOT_REACHED();
}

void ResourceHandle::receivedChallengeRejection(const AuthenticationChallenge&)
{
    ASSERT_NOT_REACHED();
}
#endif

} // namespace WebCore
