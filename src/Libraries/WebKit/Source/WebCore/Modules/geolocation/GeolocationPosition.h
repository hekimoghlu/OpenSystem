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

#include "EpochTimeStamp.h"
#include "GeolocationCoordinates.h"
#include <wtf/RefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class GeolocationPosition : public RefCounted<GeolocationPosition> {
public:
    static Ref<GeolocationPosition> create(Ref<GeolocationCoordinates>&& coordinates, EpochTimeStamp timestamp)
    {
        return adoptRef(*new GeolocationPosition(WTFMove(coordinates), timestamp));
    }

    Ref<GeolocationPosition> isolatedCopy() const
    {
        return create(m_coordinates->isolatedCopy(), m_timestamp);
    }

    EpochTimeStamp timestamp() const { return m_timestamp; }
    const GeolocationCoordinates& coords() const { return m_coordinates.get(); }
    
private:
    GeolocationPosition(Ref<GeolocationCoordinates>&& coordinates, EpochTimeStamp timestamp)
        : m_coordinates(WTFMove(coordinates))
        , m_timestamp(timestamp)
    {
    }

    Ref<GeolocationCoordinates> m_coordinates;
    EpochTimeStamp m_timestamp;
};
    
} // namespace WebCore
