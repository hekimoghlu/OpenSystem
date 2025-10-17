/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 20, 2025.
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

#if HAVE(APP_SSO)

#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS SOAuthorization;
OBJC_CLASS WKSOAuthorizationDelegate;

namespace API {
class NavigationAction;
class PageConfiguration;
}

namespace WebCore {
class ResourceRequest;
}

namespace WebKit {

class WebPageProxy;

class SOAuthorizationCoordinator {
    WTF_MAKE_TZONE_ALLOCATED(SOAuthorizationCoordinator);
    WTF_MAKE_NONCOPYABLE(SOAuthorizationCoordinator);
public:
    SOAuthorizationCoordinator();

    // For Navigation interception.
    void tryAuthorize(Ref<API::NavigationAction>&&, WebPageProxy&, Function<void(bool)>&&);

    // For PopUp interception.
    using NewPageCallback = CompletionHandler<void(RefPtr<WebPageProxy>&&)>;
    using UIClientCallback = Function<void(Ref<API::NavigationAction>&&, NewPageCallback&&)>;
    void tryAuthorize(Ref<API::PageConfiguration>&&, Ref<API::NavigationAction>&&, WebPageProxy&, NewPageCallback&&, UIClientCallback&&);

private:
    bool canAuthorize(const URL&) const;

    RetainPtr<WKSOAuthorizationDelegate> m_soAuthorizationDelegate;
    bool m_hasAppSSO { false };
};

} // namespace WebKit

#endif
