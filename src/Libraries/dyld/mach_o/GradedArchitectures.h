/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 17, 2024.
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
#ifndef mach_o_GradedArchitectures_h
#define mach_o_GradedArchitectures_h

#include <stdint.h>

#include <span>

#include "MachODefines.h"
#include "Architecture.h"


namespace mach_o {

/*!
 * @class GradedArchitectures
 *
 * @abstract
 *      Encapsulates a prioritized list of architectures
 *      Used to select slice from a fat file.
 *      Never dynamically constructed. Instead one of the existing static members is used.
 */
class VIS_HIDDEN GradedArchitectures {
public:

    bool                    hasCompatibleSlice(std::span<const Architecture> slices, bool isOSBinary, uint32_t& bestSliceIndex) const;
    bool                    isCompatible(Architecture arch, bool isOSBinary=false) const;

    bool                    checksOSBinary() const { return _requiresOSBinaries; }

    static const GradedArchitectures&  currentLaunch(const char* simArches); // for emulating how the kernel chooses which slice to exec()
    static const GradedArchitectures&  currentLoad(bool keysOff, bool platformBinariesOnly);

    // pre-built objects for use by dyld to see if a slice is loadable
    static constinit const GradedArchitectures load_mac;
    static constinit const GradedArchitectures load_macHaswell;
    static constinit const GradedArchitectures load_arm64;                 // iPhone 8 (no PAC)
    static constinit const GradedArchitectures load_arm64e;                // AppleSilicon or A12 and later
    static constinit const GradedArchitectures load_arm64e_keysOff;        // running AppStore app with keys disabled
    static constinit const GradedArchitectures load_arm64e_osBinaryOnly;   // don't load third party arm64e code
    static constinit const GradedArchitectures load_watchSeries3;
    static constinit const GradedArchitectures load_watchSeries4;

    // pre-built objects for use to see if a program is launchable
    static constinit const GradedArchitectures launch_iOS;                // arm64e iOS
    static constinit const GradedArchitectures launch_mac;                // Intel macs
    static constinit const GradedArchitectures launch_macHaswell;         // Intel macs with haswell cpu
    static constinit const GradedArchitectures launch_macAppleSilicon;    // Apple Silicon macs
    static constinit const GradedArchitectures launch_sim;                // iOS simulator for Intel macs
    static constinit const GradedArchitectures launch_simAppleSilicon;    // iOS simulator for Apple Silicon macs

private:

        constexpr GradedArchitectures(const Architecture* const a[], uint32_t size, bool requiresOSBinaries=false)
                    : _archs(a), _archCount(size/sizeof(Architecture)), _requiresOSBinaries(requiresOSBinaries) { }
                  GradedArchitectures(const GradedArchitectures&) = delete;

    // Note: this is structured so that the static members (e.g. load_arm64) can be statically built (no initializer)
    const Architecture* const* _archs;
    uint32_t            const  _archCount;
    bool                const  _requiresOSBinaries;
};

} // namespace mach_o

#endif /* mach_o_GradedArchitectures_h */
