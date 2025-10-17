/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 1, 2023.
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

#include "GeolocationClient.h"
#include "GeolocationPositionData.h"
#include "Timer.h"
#include <wtf/HashSet.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GeolocationController;

// FIXME: this should not be in WebCore. It should be moved to WebKit.
// Provides a mock object for the geolocation client.
class GeolocationClientMock : public GeolocationClient {
    WTF_MAKE_FAST_ALLOCATED;
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(GeolocationClientMock);
public:
    GeolocationClientMock();
    virtual ~GeolocationClientMock();

    void reset();
    void setController(GeolocationController*);

    void setPosition(GeolocationPositionData&&);
    void setPositionUnavailableError(const String& errorMessage);
    void setPermission(bool allowed);
    int numberOfPendingPermissionRequests() const;

    // GeolocationClient
    void geolocationDestroyed() override;
    void startUpdating(const String& authorizationToken, bool enableHighAccuracy) override;
    void stopUpdating() override;
    void setEnableHighAccuracy(bool) override;
    std::optional<GeolocationPositionData> lastPosition() override;
    void requestPermission(Geolocation&) override;
    void cancelPermissionRequest(Geolocation&) override;

private:
    void asyncUpdateController();
    void controllerTimerFired();

    void asyncUpdatePermission();
    void permissionTimerFired();

    void clearError();

    GeolocationController* m_controller;
    std::optional<GeolocationPositionData> m_lastPosition;
    bool m_hasError;
    String m_errorMessage;
    Timer m_controllerTimer;
    Timer m_permissionTimer;
    bool m_isActive;

    enum PermissionState {
        PermissionStateUnset,
        PermissionStateAllowed,
        PermissionStateDenied,
    } m_permissionState;
    typedef UncheckedKeyHashSet<RefPtr<Geolocation>> GeolocationSet;
    GeolocationSet m_pendingPermission;
};

}
