/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 19, 2024.
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
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "GradedArchitectures.h"


namespace mach_o {


bool GradedArchitectures::hasCompatibleSlice(std::span<const Architecture> slices, bool isOSBinary, uint32_t& bestSliceIndex) const
{
    if ( _requiresOSBinaries && !isOSBinary )
        return false;
    // by walking in arch order, the first match we find is the best
    for (uint32_t a=0; a < _archCount; ++a ) {
        for (uint32_t s=0; s < slices.size(); ++s ) {
            if ( slices[s] == *_archs[a] ) {
                bestSliceIndex = s;
                return true;
            }
        }
    }
    return false;
}

bool GradedArchitectures::isCompatible(Architecture arch, bool isOSBinary) const
{
    if ( _requiresOSBinaries && !isOSBinary )
        return false;
    for (uint32_t i=0; i < _archCount; ++i ) {
        if ( arch == *_archs[i] )
            return true;
    }
    return false;
}

// This is all goop to allow these to be statically compiled down (no initializer needed)
static const Architecture* archs_mac[]                  = { &Architecture::x86_64 };
static const Architecture* archs_macHaswell[]           = { &Architecture::x86_64h, &Architecture::x86_64 };
static const Architecture* archs_arm64[]                = { &Architecture::arm64, &Architecture::arm64_alt };
static const Architecture* archs_arm64e[]               = { &Architecture::arm64e };
static const Architecture* archs_arm64e_keysOff[]       = { &Architecture::arm64e, &Architecture::arm64e_v1, &Architecture::arm64, &Architecture::arm64_alt };
static const Architecture* archs_watchSeries3[]         = { &Architecture::armv7k };
static const Architecture* archs_watchSeries4[]         = { &Architecture::arm64_32 };
static const Architecture* archs_AppleSilicon[]         = { &Architecture::arm64e, &Architecture::arm64, &Architecture::x86_64 };
static const Architecture* archs_iOS[]                  = { &Architecture::arm64e, &Architecture::arm64e_v1, &Architecture::arm64, &Architecture::arm64_alt};

constinit const GradedArchitectures GradedArchitectures::load_mac(                archs_mac,            sizeof(archs_mac));
constinit const GradedArchitectures GradedArchitectures::load_macHaswell(         archs_macHaswell,     sizeof(archs_macHaswell));
constinit const GradedArchitectures GradedArchitectures::load_arm64(              archs_arm64,          sizeof(archs_arm64));
constinit const GradedArchitectures GradedArchitectures::load_arm64e(             archs_arm64e,         sizeof(archs_arm64e));
constinit const GradedArchitectures GradedArchitectures::load_arm64e_keysOff(     archs_arm64e_keysOff, sizeof(archs_arm64e_keysOff));
constinit const GradedArchitectures GradedArchitectures::load_arm64e_osBinaryOnly(archs_arm64e,         sizeof(archs_arm64e), true);
constinit const GradedArchitectures GradedArchitectures::load_watchSeries3(       archs_watchSeries3,   sizeof(archs_watchSeries3));
constinit const GradedArchitectures GradedArchitectures::load_watchSeries4(       archs_watchSeries4,   sizeof(archs_watchSeries4));

// pre-built objects for use to see if a program is launchable
constinit const GradedArchitectures GradedArchitectures::launch_iOS(            archs_iOS,          sizeof(archs_iOS));
constinit const GradedArchitectures GradedArchitectures::launch_mac(            archs_mac,          sizeof(archs_mac));
constinit const GradedArchitectures GradedArchitectures::launch_macHaswell(     archs_macHaswell,   sizeof(archs_macHaswell));
constinit const GradedArchitectures GradedArchitectures::launch_macAppleSilicon(archs_AppleSilicon, sizeof(archs_AppleSilicon));
constinit const GradedArchitectures GradedArchitectures::launch_sim(            archs_mac,          sizeof(archs_mac));
constinit const GradedArchitectures GradedArchitectures::launch_simAppleSilicon(archs_arm64,        sizeof(archs_arm64));


} // namespace mach_o





