/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 3, 2024.
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

#include "WebPage.h"
#include <WebCore/GeolocationClient.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebKit {

class WebGeolocationClient final : public WebCore::GeolocationClient {
    WTF_MAKE_TZONE_ALLOCATED(WebGeolocationClient);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(WebGeolocationClient);
public:
    WebGeolocationClient(WebPage& page)
        : m_page(page)
    {
    }

    virtual ~WebGeolocationClient();

private:
    void geolocationDestroyed() final;

    void startUpdating(const String& authorizationToken, bool needsHighAccuracy) final;
    void stopUpdating() final;
    void revokeAuthorizationToken(const String&) final;
    void setEnableHighAccuracy(bool) final;

    std::optional<WebCore::GeolocationPositionData> lastPosition() final;

    void requestPermission(WebCore::Geolocation&) final;
    void cancelPermissionRequest(WebCore::Geolocation&) final;

    WeakRef<WebPage> m_page;
};

} // namespace WebKit
