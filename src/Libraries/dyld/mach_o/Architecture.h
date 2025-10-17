/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 27, 2024.
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
#ifndef mach_o_Architecture_h
#define mach_o_Architecture_h

#include <stdint.h>
#include <string_view>

#include <mach/machine.h>
#include <mach-o/loader.h>
#include <mach-o/fat.h>

#include "MachODefines.h"

namespace mach_o {

/*!
 * @class Architecture
 *
 * @abstract
 *      Encapsulates <cpu-type,cpu-subtype> pair.
 *      Has methods to convert to and from architecture name string.
 */
class VIS_HIDDEN Architecture
{
public:
                  constexpr Architecture(cpu_type_t type, cpu_subtype_t subtype) :  _cputype(type), _cpusubtype(subtype) { }
                            Architecture(const mach_header*);
                            Architecture(const fat_arch*);
                            Architecture(const fat_arch_64*);
                  constexpr Architecture() : _cputype(0), _cpusubtype(0) { }
                  constexpr Architecture(const Architecture& other) : _cputype(other._cputype), _cpusubtype(other._cpusubtype) { }

    bool                    operator==(const Architecture& other) const;
    bool                    operator!=(const Architecture& other) const { return !operator==(other); }

    cpu_type_t              cpuType() const { return _cputype; }
    cpu_subtype_t           cpuSubtype() const { return _cpusubtype; }
    bool                    sameCpu(const Architecture& other) const { return (_cputype == other._cputype); }
    bool                    is64() const;
    bool                    isBigEndian() const;
    const char*             name() const; // returns static string
    void                    set(mach_header&) const;
    void                    set(fat_arch&) const;
    void                    set(fat_arch_64&) const;
    bool                    usesArm64Instructions() const;
    bool                    usesArm64AuthPointers() const;
    bool                    usesx86_64Instructions() const;
    bool                    usesArm32Instructions() const;
    bool                    usesThumbInstructions() const;
    bool                    usesArmZeroCostExceptions() const;
    bool                    isArm64eKernel() const;
    int                     arm64eABIVersion() const;

    static Architecture current();
    static Architecture byName(std::string_view name);

    // prebuilt values for known architectures
    static constinit const Architecture ppc;
    static constinit const Architecture i386;
    static constinit const Architecture x86_64;
    static constinit const Architecture x86_64h;
    static constinit const Architecture arm64;
    static constinit const Architecture arm64e;
    static constinit const Architecture arm64e_v1;  // wrong ABI version (for testing)
    static constinit const Architecture arm64e_old; // pre-ABI versioned (for testing)
    static constinit const Architecture arm64e_kernel;
    static constinit const Architecture arm64e_kernel_v1; // wrong ABI version (for testing)
    static constinit const Architecture arm64e_kernel_v2; // wrong ABI version (for testing)
    static constinit const Architecture arm64_32;
    static constinit const Architecture armv6;
    static constinit const Architecture armv6m;
    static constinit const Architecture armv7k;
    static constinit const Architecture armv7s;
    static constinit const Architecture armv7;
    static constinit const Architecture armv7m;
    static constinit const Architecture armv7em;
    static constinit const Architecture arm64_alt;
    static constinit const Architecture arm64_32_alt;
    static constinit const Architecture invalid;

private:
    cpu_type_t      _cputype;
    cpu_subtype_t   _cpusubtype;
};




} // namespace mach_o

#endif /* mach_o_Architecture_h */
