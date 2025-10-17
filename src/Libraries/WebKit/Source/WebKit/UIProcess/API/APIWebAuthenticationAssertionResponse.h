/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 31, 2021.
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

#if ENABLE(WEB_AUTHN)

#include "APIObject.h"
#include <WebCore/AuthenticatorAssertionResponse.h>

namespace API {

class Data;

class WebAuthenticationAssertionResponse final : public ObjectImpl<Object::Type::WebAuthenticationAssertionResponse> {
public:
    static Ref<WebAuthenticationAssertionResponse> create(Ref<WebCore::AuthenticatorAssertionResponse>&&);
    ~WebAuthenticationAssertionResponse();

    const WTF::String& name() const { return m_response->name(); }
    const WTF::String& displayName() const { return m_response->displayName(); }
    RefPtr<Data> userHandle() const;
    bool synchronizable() const { return m_response->synchronizable(); }
    const WTF::String& group() const { return m_response->group(); }
    RefPtr<Data> credentialID() const;
    const WTF::String& accessGroup() const { return m_response->accessGroup(); }

    void setLAContext(LAContext *context) { m_response->setLAContext(context); }

    WebCore::AuthenticatorAssertionResponse* response() { return m_response.ptr(); }

private:
    WebAuthenticationAssertionResponse(Ref<WebCore::AuthenticatorAssertionResponse>&&);

    Ref<WebCore::AuthenticatorAssertionResponse> m_response;
};

} // namespace API

SPECIALIZE_TYPE_TRAITS_API_OBJECT(WebAuthenticationAssertionResponse);

#endif // ENABLE(WEB_AUTHN)
