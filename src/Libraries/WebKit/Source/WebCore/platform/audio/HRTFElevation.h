/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 31, 2022.
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
#ifndef HRTFElevation_h
#define HRTFElevation_h

#include "HRTFKernel.h"
#include <memory>
#include <wtf/Noncopyable.h>
#include <wtf/RefPtr.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

// HRTFElevation contains all of the HRTFKernels (one left ear and one right ear per azimuth angle) for a particular elevation.

class HRTFElevation final {
    WTF_MAKE_TZONE_ALLOCATED(HRTFElevation);
    WTF_MAKE_NONCOPYABLE(HRTFElevation);
public:
    HRTFElevation(std::unique_ptr<HRTFKernelList> kernelListL, std::unique_ptr<HRTFKernelList> kernelListR, int elevation, float sampleRate)
        : m_kernelListL(WTFMove(kernelListL))
        , m_kernelListR(WTFMove(kernelListR))
        , m_elevationAngle(elevation)
        , m_sampleRate(sampleRate)
    {
    }

    // Loads and returns an HRTFElevation with the given HRTF database subject name and elevation from browser (or WebKit.framework) resources.
    // Normally, there will only be a single HRTF database set, but this API supports the possibility of multiple ones with different names.
    // Interpolated azimuths will be generated based on InterpolationFactor.
    // Valid values for elevation are -45 -> +90 in 15 degree increments.
    static std::unique_ptr<HRTFElevation> createForSubject(const String& subjectName, int elevation, float sampleRate);

    static void clearCache();

    // Given two HRTFElevations, and an interpolation factor x: 0 -> 1, returns an interpolated HRTFElevation.
    static std::unique_ptr<HRTFElevation> createByInterpolatingSlices(HRTFElevation* hrtfElevation1, HRTFElevation* hrtfElevation2, float x, float sampleRate);

    // Returns the list of left or right ear HRTFKernels for all the azimuths going from 0 to 360 degrees.
    HRTFKernelList* kernelListL() { return m_kernelListL.get(); }
    HRTFKernelList* kernelListR() { return m_kernelListR.get(); }

    double elevationAngle() const { return m_elevationAngle; }
    unsigned numberOfAzimuths() const { return NumberOfTotalAzimuths; }
    float sampleRate() const { return m_sampleRate; }
    
    // Returns the left and right kernels for the given azimuth index.
    // The interpolated delays based on azimuthBlend: 0 -> 1 are returned in frameDelayL and frameDelayR.
    void getKernelsFromAzimuth(double azimuthBlend, unsigned azimuthIndex, HRTFKernel* &kernelL, HRTFKernel* &kernelR, double& frameDelayL, double& frameDelayR);
    
    // Spacing, in degrees, between every azimuth loaded from resource.
    static constexpr unsigned AzimuthSpacing { 15 };
    
    // Number of azimuths loaded from resource.
    static constexpr unsigned NumberOfRawAzimuths { 360 / AzimuthSpacing };

    // Interpolates by this factor to get the total number of azimuths from every azimuth loaded from resource.
    static constexpr unsigned InterpolationFactor { 8 };
    
    // Total number of azimuths after interpolation.
    static constexpr unsigned NumberOfTotalAzimuths { NumberOfRawAzimuths * InterpolationFactor };

    // Given a specific azimuth and elevation angle, returns the left and right HRTFKernel.
    // Valid values for azimuth are 0 -> 345 in 15 degree increments.
    // Valid values for elevation are -45 -> +90 in 15 degree increments.
    // Returns true on success.
    static bool calculateKernelsForAzimuthElevation(int azimuth, int elevation, float sampleRate, const String& subjectName,
                                                    RefPtr<HRTFKernel>& kernelL, RefPtr<HRTFKernel>& kernelR);

private:
    std::unique_ptr<HRTFKernelList> m_kernelListL;
    std::unique_ptr<HRTFKernelList> m_kernelListR;
    double m_elevationAngle;
    float m_sampleRate;
};

} // namespace WebCore

#endif // HRTFElevation_h
