/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 1, 2023.
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

#include "APIObject.h"
#include "GeolocationIdentifier.h"
#include <wtf/Function.h>
#include <wtf/WeakPtr.h>

namespace WebKit {

class GeolocationPermissionRequestManagerProxy;
class WebProcessProxy;

class GeolocationPermissionRequestProxy : public RefCounted<GeolocationPermissionRequestProxy> {
public:
    static Ref<GeolocationPermissionRequestProxy> create(GeolocationPermissionRequestManagerProxy& manager, GeolocationIdentifier geolocationID, WebProcessProxy& process)
    {
        return adoptRef(*new GeolocationPermissionRequestProxy(manager, geolocationID, process));
    }

    void allow();
    void deny();
    
    void invalidate();

    WebProcessProxy* process() const;

private:
    GeolocationPermissionRequestProxy(GeolocationPermissionRequestManagerProxy&, GeolocationIdentifier, WebProcessProxy&);

    WeakPtr<GeolocationPermissionRequestManagerProxy> m_manager;
    GeolocationIdentifier m_geolocationID;
    WeakPtr<WebProcessProxy> m_process;
};

class GeolocationPermissionRequest : public API::ObjectImpl<API::Object::Type::GeolocationPermissionRequest> {
public:
    static Ref<GeolocationPermissionRequest> create(Function<void(bool)>&& completionHandler)
    {
        return adoptRef(*new GeolocationPermissionRequest(WTFMove(completionHandler)));
    }
    
    void allow() { m_completionHandler(true); }
    void deny() { m_completionHandler(false); }

private:
    GeolocationPermissionRequest(Function<void(bool)>&& completionHandler)
        : m_completionHandler(WTFMove(completionHandler))
    { }
    
    Function<void(bool)> m_completionHandler;
};

} // namespace WebKit
