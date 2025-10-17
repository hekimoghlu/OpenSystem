/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 17, 2024.
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
#ifndef mach_o_Universal_h
#define mach_o_Universal_h

#include <stdint.h>
#include <mach-o/fat.h>

#include <span>

#include "Architecture.h"
#include "GradedArchitectures.h"
#include "Header.h"

namespace mach_o {


/*!
 * @class Universal
 *
 * @abstract
 *      Abstraction for fat files
 */
struct VIS_HIDDEN Universal
{
    struct Slice
    {
        Architecture arch;
        std::span<const uint8_t> buffer;
    };

    // for examining universal files
    static const Universal* isUniversal(std::span<const uint8_t> fileContent);
    Error                   valid(uint64_t fileSize) const;
    void                    forEachSlice(void (^callback)(Slice slice, bool& stop)) const;
    bool                    bestSlice(const GradedArchitectures& ga, bool osBinary, Slice& slice) const;
    const char*             archNames(char strBuf[256]) const;
    const char*             archAndPlatformNames(char strBuf[512]) const;

protected:
    Error                   validSlice(Architecture sliceArch, uint64_t sliceOffset, uint64_t sliceLen) const;
    void                    forEachSlice(void (^callback)(Architecture arch, uint64_t sliceOffset, uint64_t sliceSize, bool& stop)) const;

                            Universal();
    void                    addMachO(const Header*);
    enum { kMaxSliceCount = 16 };

protected:
    alignas(4096) fat_header   fh;
};


} // namespace mach_o

#endif /* mach_o_Universal_h */
