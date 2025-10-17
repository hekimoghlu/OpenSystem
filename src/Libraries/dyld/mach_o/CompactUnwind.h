/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 27, 2024.
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
#ifndef mach_o_CompactUnwind_h
#define mach_o_CompactUnwind_h

#include <span>
#include <stdint.h>

#include "MachODefines.h"
#include "Error.h"
#include "Architecture.h"

namespace mach_o {

/*!
 * @class CompactUnwind
 *
 * @abstract
 *      Abstraction `__TEXT,__unwind_info` section
 */
class VIS_HIDDEN CompactUnwind
{
public:
                        // construct from a mach-o __TEXT,__unwind_info section
                        CompactUnwind(Architecture, const uint8_t* start, size_t size);

    struct UnwindInfo { uint32_t funcOffset; uint32_t encoding=0; uint32_t lsdaOffset=0; uint32_t personalityOffset=0; };
    Error               valid() const;
    void                forEachUnwindInfo(void (^callback)(const UnwindInfo&)) const;
    bool                findUnwindInfo(uint32_t funcOffset, UnwindInfo& info) const;
    void                encodingToString(uint32_t encoding, const void* funcBytes, char strBuf[128]) const;

    static uint32_t     compactUnwindEntrySize(bool is64);

protected:
                        // used by the CompactUnwindWriter subclass
                        CompactUnwind() = default;

private:
    Error               forEachFirstLevelTableEntry(void (^callback)(uint32_t funcsStartOffset, uint32_t funcsEndOffset, uint32_t secondLevelOffset, uint32_t lsdaIndexOffset)) const;
    Error               forEachSecondLevelRegularTableEntry(const struct unwind_info_regular_second_level_page_header*, void (^callback)(const UnwindInfo&)) const;
    Error               forEachSecondLevelCompressedTableEntry(const struct unwind_info_compressed_second_level_page_header*, uint32_t pageFunsOffset, void (^callback)(const UnwindInfo&)) const;
    void                encodingToString_arm64(uint32_t encoding, const void* funcBytes, char strBuf[128]) const;
    void                encodingToString_x86_64(uint32_t encoding, const void* funcBytes, char strBuf[128]) const;
    uint32_t            findLSDA(uint32_t funcOffset) const;

protected:
    Architecture                                _arch;
    const struct unwind_info_section_header*    _unwindTable      = nullptr;
    size_t                                      _unwindTableSize  = 0;
};


} // namespace mach_o

#endif // mach_o_CompactUnwind_h
