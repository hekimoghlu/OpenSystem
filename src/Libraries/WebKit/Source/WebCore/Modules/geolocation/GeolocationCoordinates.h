/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 5, 2025.
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

#include "GeolocationPositionData.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class GeolocationCoordinates : public RefCounted<GeolocationCoordinates> {
public:
    static Ref<GeolocationCoordinates> create(GeolocationPositionData&& position)
    {
        return adoptRef(*new GeolocationCoordinates(WTFMove(position)));
    }

    Ref<GeolocationCoordinates> isolatedCopy() const
    {
        return GeolocationCoordinates::create(GeolocationPositionData(m_position));
    }

    double latitude() const { return m_position.latitude; }
    double longitude() const { return m_position.longitude; }
    std::optional<double> altitude() const { return m_position.altitude; }
    double accuracy() const { return m_position.accuracy; }
    std::optional<double> altitudeAccuracy() const { return m_position.altitudeAccuracy; }
    std::optional<double> heading() const { return m_position.heading; }
    std::optional<double> speed() const { return m_position.speed; }
    std::optional<double> floorLevel() const { return m_position.floorLevel; }
    
private:
    explicit GeolocationCoordinates(GeolocationPositionData&&);

    GeolocationPositionData m_position;
};
    
} // namespace WebCore
