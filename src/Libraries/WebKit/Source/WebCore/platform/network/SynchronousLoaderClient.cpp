/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 6, 2024.
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
#include "SynchronousLoaderClient.h"

#include "AuthenticationChallenge.h"
#include "ResourceHandle.h"
#include "ResourceRequest.h"
#include "SharedBuffer.h"
#include <wtf/CompletionHandler.h>

namespace WebCore {

SynchronousLoaderClient::SynchronousLoaderClient()
    : m_messageQueue(SynchronousLoaderMessageQueue::create()) { }

SynchronousLoaderClient::~SynchronousLoaderClient() = default;

void SynchronousLoaderClient::willSendRequestAsync(ResourceHandle* handle, ResourceRequest&& request, ResourceResponse&&, CompletionHandler<void(ResourceRequest&&)>&& completionHandler)
{
    // FIXME: This needs to be fixed to follow the redirect correctly even for cross-domain requests.
    if (protocolHostAndPortAreEqual(handle->firstRequest().url(), request.url())) {
        completionHandler(WTFMove(request));
        return;
    }

    ASSERT(m_error.isNull());
    m_error = platformBadResponseError();
    completionHandler({ });
}

bool SynchronousLoaderClient::shouldUseCredentialStorage(ResourceHandle*)
{
    // FIXME: We should ask LocalFrameLoaderClient whether using credential storage is globally forbidden.
    return m_allowStoredCredentials;
}

#if USE(PROTECTION_SPACE_AUTH_CALLBACK)
void SynchronousLoaderClient::canAuthenticateAgainstProtectionSpaceAsync(ResourceHandle*, const ProtectionSpace&, CompletionHandler<void(bool)>&& completionHandler)
{
    // FIXME: We should ask LocalFrameLoaderClient. <http://webkit.org/b/65196>
    completionHandler(true);
}
#endif

void SynchronousLoaderClient::didReceiveResponseAsync(ResourceHandle*, ResourceResponse&& response, CompletionHandler<void()>&& completionHandler)
{
    m_response = WTFMove(response);
    completionHandler();
}

void SynchronousLoaderClient::didReceiveData(ResourceHandle*, const SharedBuffer& buffer, int /*encodedDataLength*/)
{
    m_data.append(buffer.span());
}

void SynchronousLoaderClient::didFinishLoading(ResourceHandle* handle, const NetworkLoadMetrics&)
{
    m_messageQueue->kill();
#if PLATFORM(COCOA)
    if (handle)
        handle->releaseDelegate();
#else
    UNUSED_PARAM(handle);
#endif
}

void SynchronousLoaderClient::didFail(ResourceHandle* handle, const ResourceError& error)
{
    ASSERT(m_error.isNull());

    m_error = error;
    
    m_messageQueue->kill();
#if PLATFORM(COCOA)
    if (handle)
        handle->releaseDelegate();
#else
    UNUSED_PARAM(handle);
#endif
}

#if USE(SOUP) || USE(CURL)
void SynchronousLoaderClient::didReceiveAuthenticationChallenge(ResourceHandle*, const AuthenticationChallenge&)
{
    ASSERT_NOT_REACHED();
}

ResourceError SynchronousLoaderClient::platformBadResponseError()
{
    ASSERT_NOT_REACHED();
    return { };
}
#endif

}
