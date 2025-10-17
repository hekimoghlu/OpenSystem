/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 19, 2025.
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
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class DeviceMotionData : public RefCounted<DeviceMotionData> {
public:
    class Acceleration : public RefCounted<DeviceMotionData::Acceleration> {
    public:
        static Ref<Acceleration> create()
        {
            return adoptRef(*new Acceleration);
        }
        static Ref<Acceleration> create(std::optional<double> x, std::optional<double> y, std::optional<double> z)
        {
            return adoptRef(*new Acceleration(x, y, z));
        }

        std::optional<double> x() const { return m_x; }
        std::optional<double> y() const { return m_y; }
        std::optional<double> z() const { return m_z; }

    private:
        Acceleration() = default;
        Acceleration(std::optional<double> x, std::optional<double> y, std::optional<double> z)
            : m_x(x)
            , m_y(y)
            , m_z(z)
        {
        }

        std::optional<double> m_x;
        std::optional<double> m_y;
        std::optional<double> m_z;
    };

    class RotationRate : public RefCounted<DeviceMotionData::RotationRate> {
    public:
        static Ref<RotationRate> create()
        {
            return adoptRef(*new RotationRate);
        }
        static Ref<RotationRate> create(std::optional<double> alpha, std::optional<double> beta, std::optional<double> gamma)
        {
            return adoptRef(*new RotationRate(alpha, beta, gamma));
        }

        std::optional<double> alpha() const { return m_alpha; }
        std::optional<double> beta() const { return m_beta; }
        std::optional<double> gamma() const { return m_gamma; }

    private:
        RotationRate() = default;
        RotationRate(std::optional<double> alpha, std::optional<double> beta, std::optional<double> gamma)
            : m_alpha(alpha)
            , m_beta(beta)
            , m_gamma(gamma)
        {
        }

        std::optional<double> m_alpha;
        std::optional<double> m_beta;
        std::optional<double> m_gamma;
    };

    WEBCORE_EXPORT static Ref<DeviceMotionData> create();
    WEBCORE_EXPORT static Ref<DeviceMotionData> create(RefPtr<Acceleration>&&, RefPtr<Acceleration>&& accelerationIncludingGravity, RefPtr<RotationRate>&&, std::optional<double> interval);

    const Acceleration* acceleration() const { return m_acceleration.get(); }
    const Acceleration* accelerationIncludingGravity() const { return m_accelerationIncludingGravity.get(); }
    const RotationRate* rotationRate() const { return m_rotationRate.get(); }
    
    std::optional<double> interval() const { return m_interval; }

private:
    DeviceMotionData() = default;
    DeviceMotionData(RefPtr<Acceleration>&&, RefPtr<Acceleration>&& accelerationIncludingGravity, RefPtr<RotationRate>&&, std::optional<double> interval);

    RefPtr<Acceleration> m_acceleration;
    RefPtr<Acceleration> m_accelerationIncludingGravity;
    RefPtr<RotationRate> m_rotationRate;
    std::optional<double> m_interval;
};

} // namespace WebCore
