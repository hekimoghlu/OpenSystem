/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 28, 2021.
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
#import "config.h"
#import "GeolocationPosition.h"

#import <CoreLocation/CLLocation.h>

namespace WebCore {

GeolocationPositionData::GeolocationPositionData(CLLocation *location)
    : timestamp(location.timestamp.timeIntervalSince1970)
    , latitude(location.coordinate.latitude)
    , longitude(location.coordinate.longitude)
    , accuracy(location.horizontalAccuracy)
{
    if (location.verticalAccuracy >= 0.0) {
        altitude = location.altitude;
        altitudeAccuracy = location.verticalAccuracy;
    }
    if (location.speed >= 0.0)
        speed = location.speed;
    if (location.course >= 0.0)
        heading = location.course;
#if !PLATFORM(MACCATALYST)
    if (location.floor)
        floorLevel = location.floor.level;
#endif
}

} // namespace WebCore
