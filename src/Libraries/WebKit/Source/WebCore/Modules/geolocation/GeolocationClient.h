/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 28, 2024.
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

#include <optional>
#include <wtf/CheckedPtr.h>
#include <wtf/Forward.h>

namespace WebCore {

class Geolocation;
class GeolocationPositionData;
class Page;

class GeolocationClient : public CanMakeCheckedPtr<GeolocationClient> {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(GeolocationClient);
public:
    virtual void geolocationDestroyed() = 0;

    virtual void startUpdating(const String& authorizationToken, bool needsHighAccuracy) = 0;
    virtual void stopUpdating() = 0;
    virtual void revokeAuthorizationToken(const String&) { }

    // FIXME: The V2 Geolocation specification proposes that this property is
    // renamed. See http://www.w3.org/2008/geolocation/track/issues/6
    // We should update WebKit to reflect this if and when the V2 specification
    // is published.
    virtual void setEnableHighAccuracy(bool) = 0;
    virtual std::optional<GeolocationPositionData> lastPosition() = 0;

    virtual void requestPermission(Geolocation&) = 0;
    virtual void cancelPermissionRequest(Geolocation&) = 0;

    void provideGeolocationTo(Page*, GeolocationClient&);

protected:
    virtual ~GeolocationClient() = default;
};

WEBCORE_EXPORT void provideGeolocationTo(Page*, GeolocationClient&);

} // namespace WebCore
