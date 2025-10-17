/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 7, 2024.
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

#include "WebAuthenticationFlags.h"
#include <wtf/CompletionHandler.h>
#include <wtf/HashSet.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/spi/cocoa/SecuritySPI.h>
#include <wtf/text/WTFString.h>

OBJC_CLASS LAContext;

namespace WebCore {
class AuthenticatorAssertionResponse;
}

namespace API {

class WebAuthenticationPanelClient : public RefCounted<WebAuthenticationPanelClient> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(WebAuthenticationPanelClient);
public:
    static Ref<WebAuthenticationPanelClient> create() { return adoptRef(*new WebAuthenticationPanelClient); }
    virtual ~WebAuthenticationPanelClient() = default;

    virtual void updatePanel(WebKit::WebAuthenticationStatus) const { }
    virtual void dismissPanel(WebKit::WebAuthenticationResult) const { }
    virtual void requestPin(uint64_t, CompletionHandler<void(const WTF::String&)>&& completionHandler) const { completionHandler(WTF::String()); }
    virtual void requestNewPin(uint64_t, CompletionHandler<void(const WTF::String&)>&& completionHandler) const { completionHandler(WTF::String()); }
    virtual void selectAssertionResponse(Vector<Ref<WebCore::AuthenticatorAssertionResponse>>&&, WebKit::WebAuthenticationSource, CompletionHandler<void(WebCore::AuthenticatorAssertionResponse*)>&& completionHandler) const { completionHandler(nullptr); }
    virtual void decidePolicyForLocalAuthenticator(CompletionHandler<void(WebKit::LocalAuthenticatorPolicy)>&& completionHandler) const { completionHandler(WebKit::LocalAuthenticatorPolicy::Disallow); }
    virtual void requestLAContextForUserVerification(CompletionHandler<void(LAContext *)>&& completionHandler) const { completionHandler(nullptr); }

protected:
    WebAuthenticationPanelClient() = default;
};

} // namespace API

#endif // ENABLE(WEB_AUTHN)
