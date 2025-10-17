/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 14, 2025.
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

#if ENABLE(WEB_AUDIO)

#include "HRTFDatabase.h"

#include "HRTFElevation.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(HRTFDatabase);

HRTFDatabase::HRTFDatabase(float sampleRate)
    : m_sampleRate(sampleRate)
{
    unsigned elevationIndex = 0;
    for (int elevation = MinElevation; elevation <= MaxElevation; elevation += RawElevationAngleSpacing) {
        std::unique_ptr<HRTFElevation> hrtfElevation = HRTFElevation::createForSubject("Composite"_s, elevation, sampleRate);
        ASSERT(hrtfElevation.get());
        if (!hrtfElevation.get())
            return;
        
        m_elevations[elevationIndex] = WTFMove(hrtfElevation);
        elevationIndex += InterpolationFactor;
    }

    // Now, go back and interpolate elevations.
    if (InterpolationFactor > 1) {
        for (unsigned i = 0; i < NumberOfTotalElevations; i += InterpolationFactor) {
            unsigned j = (i + InterpolationFactor);
            if (j >= NumberOfTotalElevations)
                j = i; // for last elevation interpolate with itself

            // Create the interpolated convolution kernels and delays.
            for (unsigned jj = 1; jj < InterpolationFactor; ++jj) {
                float x = static_cast<float>(jj) / static_cast<float>(InterpolationFactor);
                m_elevations[i + jj] = HRTFElevation::createByInterpolatingSlices(m_elevations[i].get(), m_elevations[j].get(), x, sampleRate);
                ASSERT(m_elevations[i + jj].get());
            }
        }
    }
}

void HRTFDatabase::getKernelsFromAzimuthElevation(double azimuthBlend, unsigned azimuthIndex, double elevationAngle, HRTFKernel* &kernelL, HRTFKernel* &kernelR,
                                                  double& frameDelayL, double& frameDelayR)
{
    unsigned elevationIndex = indexFromElevationAngle(elevationAngle);
    ASSERT_WITH_SECURITY_IMPLICATION(elevationIndex < m_elevations.size() && m_elevations.size() > 0);
    
    if (!m_elevations.size()) {
        kernelL = 0;
        kernelR = 0;
        return;
    }
    
    if (elevationIndex > m_elevations.size() - 1)
        elevationIndex = m_elevations.size() - 1;    
    
    HRTFElevation* hrtfElevation = m_elevations[elevationIndex].get();
    ASSERT(hrtfElevation);
    if (!hrtfElevation) {
        kernelL = 0;
        kernelR = 0;
        return;
    }
    
    hrtfElevation->getKernelsFromAzimuth(azimuthBlend, azimuthIndex, kernelL, kernelR, frameDelayL, frameDelayR);
}                                                     

unsigned HRTFDatabase::indexFromElevationAngle(double elevationAngle)
{
    // Clamp to allowed range.
    elevationAngle = std::max(static_cast<double>(MinElevation), elevationAngle);
    elevationAngle = std::min(static_cast<double>(MaxElevation), elevationAngle);

    unsigned elevationIndex = static_cast<int>(InterpolationFactor * (elevationAngle - MinElevation) / RawElevationAngleSpacing);    
    return elevationIndex;
}

} // namespace WebCore

#endif // ENABLE(WEB_AUDIO)
