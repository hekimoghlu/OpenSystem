/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 6, 2024.
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

#include <cmath>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

#if PLATFORM(COCOA)
OBJC_CLASS CLLocation;
#endif

namespace WebCore {

class GeolocationPositionData {
public:
    GeolocationPositionData() = default;

    GeolocationPositionData(double timestamp, double latitude, double longitude, double accuracy)
        : timestamp(timestamp)
        , latitude(latitude)
        , longitude(longitude)
        , accuracy(accuracy)
    {
    }
    
    GeolocationPositionData(double timestamp, double latitude, double longitude, double accuracy, std::optional<double> altitude, std::optional<double> altitudeAccuracy, std::optional<double> heading, std::optional<double> speed, std::optional<double> floorLevel)
        : timestamp(timestamp)
        , latitude(latitude)
        , longitude(longitude)
        , accuracy(accuracy)
        , altitude(altitude)
        , altitudeAccuracy(altitudeAccuracy)
        , heading(heading)
        , speed(speed)
        , floorLevel(floorLevel)
    {
    }

#if PLATFORM(COCOA)
    WEBCORE_EXPORT explicit GeolocationPositionData(CLLocation*);
#endif

    double timestamp { std::numeric_limits<double>::quiet_NaN() };

    double latitude { std::numeric_limits<double>::quiet_NaN() };
    double longitude { std::numeric_limits<double>::quiet_NaN() };
    double accuracy { std::numeric_limits<double>::quiet_NaN() };

    std::optional<double> altitude;
    std::optional<double> altitudeAccuracy;
    std::optional<double> heading;
    std::optional<double> speed;
    std::optional<double> floorLevel;

    bool isValid() const;
};

inline bool GeolocationPositionData::isValid() const
{
    return !std::isnan(timestamp) && !std::isnan(latitude) && !std::isnan(longitude) && !std::isnan(accuracy);
}

} // namespace WebCore
