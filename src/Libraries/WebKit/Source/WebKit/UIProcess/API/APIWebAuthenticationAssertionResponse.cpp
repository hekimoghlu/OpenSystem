/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 10, 2023.
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
#include "APIWebAuthenticationAssertionResponse.h"

#if ENABLE(WEB_AUTHN)

#include "APIData.h"
#include <JavaScriptCore/ArrayBuffer.h>

namespace API {
using namespace WebCore;

Ref<WebAuthenticationAssertionResponse> WebAuthenticationAssertionResponse::create(Ref<WebCore::AuthenticatorAssertionResponse>&& response)
{
    return adoptRef(*new WebAuthenticationAssertionResponse(WTFMove(response)));
}

WebAuthenticationAssertionResponse::WebAuthenticationAssertionResponse(Ref<WebCore::AuthenticatorAssertionResponse>&& response)
    : m_response(WTFMove(response))
{
}

WebAuthenticationAssertionResponse::~WebAuthenticationAssertionResponse() = default;

RefPtr<Data> WebAuthenticationAssertionResponse::userHandle() const
{
    RefPtr<API::Data> data;
    if (RefPtr userHandle = m_response->userHandle()) {
        auto userHandleSpan = userHandle->span();
        data = API::Data::createWithoutCopying(userHandleSpan, [userHandle = WTFMove(userHandle)] { });
    }
    return data;
}

RefPtr<Data> WebAuthenticationAssertionResponse::credentialID() const
{
    RefPtr<API::Data> data;
    if (RefPtr rawId = m_response->rawId()) {
        auto rawIdSpan = rawId->span();
        data = API::Data::createWithoutCopying(rawIdSpan, [rawId = WTFMove(rawId)] { });
    }
    return data;
}

} // namespace API

#endif // ENABLE(WEB_AUTHN)
