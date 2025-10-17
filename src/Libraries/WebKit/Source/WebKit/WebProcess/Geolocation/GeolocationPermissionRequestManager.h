/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 12, 2024.
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

#include "GeolocationIdentifier.h"
#include <wtf/CheckedPtr.h>
#include <wtf/HashMap.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WeakRef.h>

namespace WebCore {
class Geolocation;
}

namespace WebKit {

class WebPage;

class GeolocationPermissionRequestManager {
    WTF_MAKE_TZONE_ALLOCATED(GeolocationPermissionRequestManager);
public:
    explicit GeolocationPermissionRequestManager(WebPage&);
    ~GeolocationPermissionRequestManager();

    void startRequestForGeolocation(WebCore::Geolocation&);
    void cancelRequestForGeolocation(WebCore::Geolocation&);
    void revokeAuthorizationToken(const String&);

    void didReceiveGeolocationPermissionDecision(GeolocationIdentifier, const String& authorizationToken);

    void ref() const;
    void deref() const;

private:
    using IDToGeolocationMap = HashMap<GeolocationIdentifier, WeakRef<WebCore::Geolocation>>;
    using GeolocationToIDMap = HashMap<WeakRef<WebCore::Geolocation>, GeolocationIdentifier>;
    IDToGeolocationMap m_idToGeolocationMap;
    GeolocationToIDMap m_geolocationToIDMap;

    Ref<WebPage> protectedPage() const;

    WeakRef<WebPage> m_page;
};

} // namespace WebKit
