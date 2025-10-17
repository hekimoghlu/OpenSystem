/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 13, 2025.
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
#ifndef Distance_h
#define Distance_h

#include <wtf/TZoneMalloc.h>

namespace WebCore {

// Distance models are defined according to the OpenAL specification:
// http://connect.creativelabs.com/openal/Documentation/OpenAL%201.1%20Specification.htm.

enum class DistanceModelType {
    Linear,
    Inverse,
    Exponential
};

class DistanceEffect final {
    WTF_MAKE_TZONE_ALLOCATED(DistanceEffect);
public:
    DistanceEffect();

    // Returns scalar gain for the given distance the current distance model is used
    double gain(double distance) const;

    DistanceModelType model() const { return m_model; }

    void setModel(DistanceModelType model, bool clamped)
    {
        m_model = model;
        m_isClamped = clamped;
    }

    // Distance params
    void setRefDistance(double refDistance) { m_refDistance = refDistance; }
    void setMaxDistance(double maxDistance) { m_maxDistance = maxDistance; }
    void setRolloffFactor(double rolloffFactor) { m_rolloffFactor = rolloffFactor; }

    double refDistance() const { return m_refDistance; }
    double maxDistance() const { return m_maxDistance; }
    double rolloffFactor() const { return m_rolloffFactor; }

protected:
    double linearGain(double distance) const;
    double inverseGain(double distance) const;
    double exponentialGain(double distance) const;

    DistanceModelType m_model { DistanceModelType::Inverse };
    bool m_isClamped { true };
    double m_refDistance { 1 };
    double m_maxDistance { 10000 };
    double m_rolloffFactor { 1 };
};

} // namespace WebCore

#endif // Distance_h
