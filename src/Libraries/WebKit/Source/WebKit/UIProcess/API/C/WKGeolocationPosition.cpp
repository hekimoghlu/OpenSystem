/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 13, 2023.
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
#include "WKGeolocationPosition.h"

#include "WKAPICast.h"
#include "WebFrameProxy.h"
#include "WebGeolocationPosition.h"

using namespace WebKit;

WKTypeID WKGeolocationPositionGetTypeID()
{
    return toAPI(WebGeolocationPosition::APIType);
}

WKGeolocationPositionRef WKGeolocationPositionCreate(double timestamp, double latitude, double longitude, double accuracy)
{
    return WKGeolocationPositionCreate_c(timestamp, latitude, longitude, accuracy, false, 0., false, 0., false, 0., false, 0., false, 0.);
}

WKGeolocationPositionRef WKGeolocationPositionCreate_b(double timestamp, double latitude, double longitude, double accuracy, bool providesAltitude, double altitude, bool providesAltitudeAccuracy, double altitudeAccuracy, bool providesHeading, double heading, bool providesSpeed, double speed)
{
    return WKGeolocationPositionCreate_c(timestamp, latitude, longitude, accuracy, providesAltitude, altitude, providesAltitudeAccuracy, altitudeAccuracy, providesHeading, heading, providesSpeed, speed, false, 0.);
}

WKGeolocationPositionRef WKGeolocationPositionCreate_c(double timestamp, double latitude, double longitude, double accuracy, bool providesAltitude, double altitude, bool providesAltitudeAccuracy, double altitudeAccuracy, bool providesHeading, double heading, bool providesSpeed, double speed, bool providesFloorLevel, double floorLevel)
{
    WebCore::GeolocationPositionData corePosition { timestamp, latitude, longitude, accuracy };
    if (providesAltitude)
        corePosition.altitude = altitude;
    if (providesAltitudeAccuracy)
        corePosition.altitudeAccuracy = altitudeAccuracy;
    if (providesHeading)
        corePosition.heading = heading;
    if (providesSpeed)
        corePosition.speed = speed;
    if (providesFloorLevel)
        corePosition.floorLevel = floorLevel;

    auto position = WebGeolocationPosition::create(WTFMove(corePosition));
    return toAPI(&position.leakRef());
}
