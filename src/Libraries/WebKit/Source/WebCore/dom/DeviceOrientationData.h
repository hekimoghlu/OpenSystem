/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 3, 2023.
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

#include <optional>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class DeviceOrientationData : public RefCounted<DeviceOrientationData> {
public:
    static Ref<DeviceOrientationData> create()
    {
        return adoptRef(*new DeviceOrientationData);
    }

#if PLATFORM(IOS_FAMILY)
    WEBCORE_EXPORT static Ref<DeviceOrientationData> create(std::optional<double> alpha, std::optional<double> beta, std::optional<double> gamma, std::optional<double> compassHeading, std::optional<double> compassAccuracy);
#else
    WEBCORE_EXPORT static Ref<DeviceOrientationData> create(std::optional<double> alpha, std::optional<double> beta, std::optional<double> gamma, std::optional<bool> absolute);
#endif

    std::optional<double> alpha() const { return m_alpha; }
    std::optional<double> beta() const { return m_beta; }
    std::optional<double> gamma() const { return m_gamma; }
#if PLATFORM(IOS_FAMILY)
    std::optional<double> compassHeading() const { return m_compassHeading; }
    std::optional<double> compassAccuracy() const { return m_compassAccuracy; }
#else
    std::optional<bool> absolute() const { return m_absolute; }
#endif

private:
    DeviceOrientationData() = default;
#if PLATFORM(IOS_FAMILY)
    DeviceOrientationData(std::optional<double> alpha, std::optional<double> beta, std::optional<double> gamma, std::optional<double> compassHeading, std::optional<double> compassAccuracy);
#else
    DeviceOrientationData(std::optional<double> alpha, std::optional<double> beta, std::optional<double> gamma, std::optional<bool> absolute);
#endif

    std::optional<double> m_alpha;
    std::optional<double> m_beta;
    std::optional<double> m_gamma;
#if PLATFORM(IOS_FAMILY)
    std::optional<double> m_compassHeading;
    std::optional<double> m_compassAccuracy;
#else
    std::optional<bool> m_absolute;
#endif
};

} // namespace WebCore
