/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 7, 2024.
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
#include "APIUIClient.h"

#include "UserMediaPermissionCheckProxy.h"
#include "UserMediaPermissionRequestProxy.h"
#include "WebPageProxy.h"
#include <wtf/TZoneMallocInlines.h>

namespace API {

WTF_MAKE_TZONE_ALLOCATED_IMPL(UIClient);

void UIClient::checkUserMediaPermissionForOrigin(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, SecurityOrigin&, SecurityOrigin&, WebKit::UserMediaPermissionCheckProxy& request)
{
    request.deny();
}

void UIClient::decidePolicyForUserMediaPermissionRequest(WebKit::WebPageProxy&, WebKit::WebFrameProxy&, SecurityOrigin&, SecurityOrigin&, WebKit::UserMediaPermissionRequestProxy& request)
{
    request.doDefaultAction();
}

void UIClient::createNewPage(WebKit::WebPageProxy&, Ref<API::PageConfiguration>&&, Ref<NavigationAction>&&, CompletionHandler<void(RefPtr<WebKit::WebPageProxy>&&)>&& completionHandler)
{
    completionHandler(nullptr);
}

void UIClient::decidePolicyForMediaKeySystemPermissionRequest(WebKit::WebPageProxy& page, SecurityOrigin& origin, const WTF::String& keySystem, CompletionHandler<void(bool)>&& completionHandler)
{
    page.requestMediaKeySystemPermissionByDefaultAction(origin.securityOrigin(), WTFMove(completionHandler));
}

} // namespace API
