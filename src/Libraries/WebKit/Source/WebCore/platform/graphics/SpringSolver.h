/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 26, 2023.
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

namespace WebCore {

class SpringSolver {
public:
    SpringSolver(double mass, double stiffness, double damping, double initialVelocity)
    {
        m_w0 = std::sqrt(stiffness / mass);
        m_zeta = damping / (2 * std::sqrt(stiffness * mass));

        if (m_zeta < 1) {
            // Under-damped.
            m_wd = m_w0 * std::sqrt(1 - m_zeta * m_zeta);
            m_A = 1;
            m_B = (m_zeta * m_w0 + -initialVelocity) / m_wd;
        } else {
            // Critically damped (ignoring over-damped case for now).
            m_A = 1;
            m_B = -initialVelocity + m_w0;
        }
    }

    double solve(double t) const
    {
        if (m_zeta < 1) {
            // Under-damped
            t = std::exp(-t * m_zeta * m_w0) * (m_A * std::cos(m_wd * t) + m_B * std::sin(m_wd * t));
        } else {
            // Critically damped (ignoring over-damped case for now).
            t = (m_A + m_B * t) * std::exp(-t * m_w0);
        }

        // Map range from [1..0] to [0..1].
        return 1 - t;
    }

private:
    double m_w0;
    double m_zeta;
    double m_wd;
    double m_A;
    double m_B;
};

} // namespace WebCore
