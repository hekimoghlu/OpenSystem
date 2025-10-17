/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 3, 2023.
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
#include "WebMediaKeySystemClient.h"

#if ENABLE(ENCRYPTED_MEDIA)

#include "MediaKeySystemPermissionRequestManager.h"
#include "WebPage.h"
#include <WebCore/MediaKeySystemController.h>
#include <WebCore/MediaKeySystemRequest.h>
#include <wtf/TZoneMallocInlines.h>

namespace WebKit {
using namespace WebCore;

WTF_MAKE_TZONE_ALLOCATED_IMPL(WebMediaKeySystemClient);

WebMediaKeySystemClient::WebMediaKeySystemClient(WebPage& page)
    : m_page(page)
{
}

void WebMediaKeySystemClient::pageDestroyed()
{
    delete this;
}

Ref<WebPage> WebMediaKeySystemClient::protectedPage() const
{
    return m_page.get();
}

void WebMediaKeySystemClient::requestMediaKeySystem(MediaKeySystemRequest& request)
{
    protectedPage()->protectedMediaKeySystemPermissionRequestManager()->startMediaKeySystemRequest(request);
}

void WebMediaKeySystemClient::cancelMediaKeySystemRequest(MediaKeySystemRequest& request)
{
    protectedPage()->protectedMediaKeySystemPermissionRequestManager()->cancelMediaKeySystemRequest(request);
}

} // namespace WebKit;

#endif // ENABLE(ENCRYPTED_MEDIA)
