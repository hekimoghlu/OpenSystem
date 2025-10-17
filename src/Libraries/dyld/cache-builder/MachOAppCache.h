/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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
#ifndef MachOAppCache_h
#define MachOAppCache_h

#include "MachOAnalyzer.h"

namespace dyld3 {

struct MachOAppCache : public MachOAnalyzer {
    // Taken from kmod.h
    enum {
        kmodMaxName = 64
    };
    #pragma pack(push, 4)
    struct KModInfo64_v1 {
        uint64_t            next_addr;
        int32_t             info_version;
        uint32_t            id;
        uint8_t             name[kmodMaxName];
        uint8_t             version[kmodMaxName];
        int32_t             reference_count;
        uint64_t            reference_list_addr;
        uint64_t            address;
        uint64_t            size;
        uint64_t            hdr_size;
        uint64_t            start_addr;
        uint64_t            stop_addr;
    };
    #pragma pack(pop)

    void forEachDylib(Diagnostics& diag, void (^callback)(const MachOAnalyzer* ma, const char* name, bool& stop)) const;

    // Walk the __PRELINK_INFO dictionary and return each bundle and its libraries
    void forEachPrelinkInfoLibrary(Diagnostics& diags,
                                   void (^callback)(const char* bundleName, const char* relativePath,
                                                    const std::vector<const char*>& deps)) const;
};

} // namespace dyld3

#endif /* MachOAppCache_h */
