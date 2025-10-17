/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 2, 2024.
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

#if ENABLE(GEOLOCATION)

#include "ActivityStateChangeObserver.h"
#include "Geolocation.h"
#include "Page.h"
#include "RegistrableDomain.h"
#include <wtf/HashSet.h>
#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class GeolocationClient;
class GeolocationError;
class GeolocationPositionData;

class GeolocationController : public Supplement<Page>, private ActivityStateChangeObserver {
    WTF_MAKE_TZONE_ALLOCATED(GeolocationController);
    WTF_MAKE_NONCOPYABLE(GeolocationController);
public:
    GeolocationController(Page&, GeolocationClient&);
    ~GeolocationController();

    void addObserver(Geolocation&, bool enableHighAccuracy);
    void removeObserver(Geolocation&);

    void requestPermission(Geolocation&);
    void cancelPermissionRequest(Geolocation&);

    WEBCORE_EXPORT void positionChanged(const std::optional<GeolocationPositionData>&);
    WEBCORE_EXPORT void errorOccurred(GeolocationError&);

    std::optional<GeolocationPositionData> lastPosition();

    GeolocationClient& client();

    WEBCORE_EXPORT static ASCIILiteral supplementName();
    static GeolocationController* from(Page* page) { return static_cast<GeolocationController*>(Supplement<Page>::from(page, supplementName())); }

    void revokeAuthorizationToken(const String&);

    void didNavigatePage();

private:
    WeakRef<Page> m_page;
    CheckedPtr<GeolocationClient> m_client; // Only becomes null in the class destructor

    void activityStateDidChange(OptionSet<ActivityState> oldActivityState, OptionSet<ActivityState> newActivityState) override;

    std::optional<GeolocationPositionData> m_lastPosition;

    bool needsHighAccuracy() const { return !m_highAccuracyObservers.isEmpty(); }

    void startUpdatingIfNecessary();
    void stopUpdatingIfNecessary();

    typedef HashSet<Ref<Geolocation>> ObserversSet;
    // All observers; both those requesting high accuracy and those not.
    ObserversSet m_observers;
    ObserversSet m_highAccuracyObservers;

    // While the page is not visible, we pend permission requests.
    HashSet<Ref<Geolocation>> m_pendingPermissionRequest;

    RegistrableDomain m_registrableDomain;
    bool m_isUpdating { false };
};

} // namespace WebCore

#endif // ENABLE(GEOLOCATION)
