/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 20, 2025.
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
#ifndef mach_o_DwarfDebug_h
#define mach_o_DwarfDebug_h

#include <span>
#include <stdint.h>

#include "MachODefines.h"
#include "Error.h"
#include "Architecture.h"

namespace mach_o {

/*!
 * @class DwarfDebug
 *
 * @abstract
 *      parses info from __DWARF sections
 */
class VIS_HIDDEN DwarfDebug
{
public:
                        // construct from a mach-o __DWARF,__debug* sections
                        DwarfDebug(std::span<const uint8_t> debugInfo, std::span<const uint8_t> abbrev,
                                   std::span<const uint8_t> strings, std::span<const uint8_t> stringOffs);
    const char*         sourceFileDir() const  { return _tuDir; }
    const char*         sourceFileName() const { return _tuFileName; }

private:
    void                parseCompilationUnit();
    const char*         getDwarfString(uint64_t form, const uint8_t*& di, bool dwarf64);
    const char*         getStrxString(uint64_t idx, bool dwarf64);
    bool                skip_form(const uint8_t*& offset, const uint8_t* end, uint64_t form, uint8_t addr_size, bool dwarf64);

    std::span<const uint8_t> _debugInfo;
    std::span<const uint8_t> _abbrev;
    std::span<const uint8_t> _strings;
    std::span<const uint8_t> _stringOffsets;
    const char*              _tuDir       = nullptr;
    const char*              _tuFileName  = nullptr;
};


} // namespace mach_o

#endif // mach_o_DwarfDebug_h
