/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 10, 2023.
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
#include "DeviceMotionData.h"

namespace WebCore {

Ref<DeviceMotionData> DeviceMotionData::create()
{
    return adoptRef(*new DeviceMotionData);
}

Ref<DeviceMotionData> DeviceMotionData::create(RefPtr<Acceleration>&& acceleration, RefPtr<Acceleration>&& accelerationIncludingGravity, RefPtr<RotationRate>&& rotationRate, std::optional<double> interval)
{
    return adoptRef(*new DeviceMotionData(WTFMove(acceleration), WTFMove(accelerationIncludingGravity), WTFMove(rotationRate), interval));
}

DeviceMotionData::DeviceMotionData(RefPtr<Acceleration>&& acceleration, RefPtr<Acceleration>&& accelerationIncludingGravity, RefPtr<RotationRate>&& rotationRate, std::optional<double> interval)
    : m_acceleration(WTFMove(acceleration))
    , m_accelerationIncludingGravity(WTFMove(accelerationIncludingGravity))
    , m_rotationRate(WTFMove(rotationRate))
    , m_interval(interval)
{
}

} // namespace WebCore
