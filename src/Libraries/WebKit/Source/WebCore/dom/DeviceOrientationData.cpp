/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 4, 2023.
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
#include "DeviceOrientationData.h"

namespace WebCore {

#if PLATFORM(IOS_FAMILY)

// FIXME: We should reconcile the iOS and OpenSource differences.
Ref<DeviceOrientationData> DeviceOrientationData::create(std::optional<double> alpha, std::optional<double> beta, std::optional<double> gamma, std::optional<double> compassHeading, std::optional<double> compassAccuracy)
{
    return adoptRef(*new DeviceOrientationData(alpha, beta, gamma, compassHeading, compassAccuracy));
}

DeviceOrientationData::DeviceOrientationData(std::optional<double> alpha, std::optional<double> beta, std::optional<double> gamma, std::optional<double> compassHeading, std::optional<double> compassAccuracy)
    : m_alpha(alpha)
    , m_beta(beta)
    , m_gamma(gamma)
    , m_compassHeading(compassHeading)
    , m_compassAccuracy(compassAccuracy)
{
}

#else

Ref<DeviceOrientationData> DeviceOrientationData::create(std::optional<double> alpha, std::optional<double> beta, std::optional<double> gamma, std::optional<bool> absolute)
{
    return adoptRef(*new DeviceOrientationData(alpha, beta, gamma, absolute));
}

DeviceOrientationData::DeviceOrientationData(std::optional<double> alpha, std::optional<double> beta, std::optional<double> gamma, std::optional<bool> absolute)
    : m_alpha(alpha)
    , m_beta(beta)
    , m_gamma(gamma)
    , m_absolute(absolute)
{
}

#endif

} // namespace WebCore
