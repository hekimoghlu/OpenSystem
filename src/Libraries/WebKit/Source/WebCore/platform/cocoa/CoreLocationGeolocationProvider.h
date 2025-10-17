/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 9, 2024.
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

#if HAVE(CORE_LOCATION)
#include <wtf/CompletionHandler.h>
#include <wtf/Forward.h>
#include <wtf/RetainPtr.h>
#include <wtf/TZoneMalloc.h>

OBJC_CLASS WebCLLocationManager;

namespace WebCore {

class GeolocationPositionData;
class RegistrableDomain;

class WEBCORE_EXPORT CoreLocationGeolocationProvider {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(CoreLocationGeolocationProvider, WEBCORE_EXPORT);
public:
    class Client {
    public:
        virtual ~Client() { }

        virtual void geolocationAuthorizationGranted(const String& /*websiteIdentifier*/) { }
        virtual void geolocationAuthorizationDenied(const String& /*websiteIdentifier*/) { }
        virtual void positionChanged(const String& websiteIdentifier, GeolocationPositionData&&) = 0;
        virtual void errorOccurred(const String& websiteIdentifier, const String& errorMessage) = 0;
        virtual void resetGeolocation(const String& websiteIdentifier) = 0;
    };

    enum class Mode : bool { AuthorizationOnly, AuthorizationAndLocationUpdates };
    CoreLocationGeolocationProvider(const RegistrableDomain&, Client&, Mode = Mode::AuthorizationAndLocationUpdates);
    ~CoreLocationGeolocationProvider();

    void setEnableHighAccuracy(bool);

    static void requestAuthorization(const RegistrableDomain&, CompletionHandler<void(bool)>&&);

private:
    RetainPtr<WebCLLocationManager> m_locationManager;
};

#endif // HAVE(CORE_LOCATION)

} // namespace WebCore
