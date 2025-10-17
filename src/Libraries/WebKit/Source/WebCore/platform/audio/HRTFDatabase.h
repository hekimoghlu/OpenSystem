/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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
#ifndef HRTFDatabase_h
#define HRTFDatabase_h

#include "HRTFElevation.h"
#include <memory>
#include <wtf/Forward.h>
#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>

namespace WebCore {

class HRTFKernel;

class HRTFDatabase final {
    WTF_MAKE_TZONE_ALLOCATED(HRTFDatabase);
    WTF_MAKE_NONCOPYABLE(HRTFDatabase);
public:
    explicit HRTFDatabase(float sampleRate);

    // getKernelsFromAzimuthElevation() returns a left and right ear kernel, and an interpolated left and right frame delay for the given azimuth and elevation.
    // azimuthBlend must be in the range 0 -> 1.
    // Valid values for azimuthIndex are 0 -> HRTFElevation::NumberOfTotalAzimuths - 1 (corresponding to angles of 0 -> 360).
    // Valid values for elevationAngle are MinElevation -> MaxElevation.
    void getKernelsFromAzimuthElevation(double azimuthBlend, unsigned azimuthIndex, double elevationAngle, HRTFKernel* &kernelL, HRTFKernel* &kernelR, double& frameDelayL, double& frameDelayR);

    // Returns the number of different azimuth angles.
    static unsigned numberOfAzimuths() { return HRTFElevation::NumberOfTotalAzimuths; }

    float sampleRate() const { return m_sampleRate; }

    // Number of elevations loaded from resource.
    static const unsigned NumberOfRawElevations { 10 };

private:
    // Minimum and maximum elevation angles (inclusive) for a HRTFDatabase.
    static constexpr int MinElevation { -45 };
    static constexpr int MaxElevation { 90 };
    static constexpr unsigned RawElevationAngleSpacing { 15 };

    // Interpolates by this factor to get the total number of elevations from every elevation loaded from resource.
    static constexpr unsigned InterpolationFactor { 1 };
    
    // Total number of elevations after interpolation.
    static constexpr unsigned NumberOfTotalElevations { NumberOfRawElevations * InterpolationFactor };

    // Returns the index for the correct HRTFElevation given the elevation angle.
    static unsigned indexFromElevationAngle(double);

    Vector<std::unique_ptr<HRTFElevation>> m_elevations { NumberOfTotalElevations };
    float m_sampleRate;
};

} // namespace WebCore

#endif // HRTFDatabase_h
