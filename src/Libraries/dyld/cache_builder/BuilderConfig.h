/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, May 25, 2022.
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
#ifndef BuilderConfig_hpp
#define BuilderConfig_hpp

#include "Timer.h"
#include "Types.h"

#include <string>

namespace cache_builder
{

constexpr uint64_t operator"" _KB(uint64_t v)
{
    return (1ULL << 10) * v;
}

constexpr uint64_t operator"" _MB(uint64_t v)
{
    return (1ULL << 20) * v;
}

constexpr uint64_t operator"" _GB(uint64_t v)
{
    return (1ULL << 30) * v;
}

constexpr uint64_t operator"" _GB(long double v)
{
    return (1ULL << 30) * v;
}

struct BuilderOptions;

// Layout handles all the different kinds of cache we can build.  They are:
//  - regular contiguous:    The cache is one big file, eg, arm64 simulators
//  - regular discontiguous: The cache is one big file, eg, x86_64 simulators
//  - large contiguous:      The cache is one or more files, which each contain TEXT/DATA/LINKEDIT.  Eg, macOS/iOS/tvOS arm64
//  - large discontiguous:   The cache is one or more files, which each contain TEXT/DATA/LINKEDIT.  Eg, macOS x86_64
struct Layout
{
    Layout(const BuilderOptions& options);

    // Used only for x86_64*
    struct Discontiguous
    {
        // For the host OS, regions should be 1GB aligned
        // If this has a value, then we use it.  Otherwise we fall back to the sim fixed addresses
        std::optional<uint64_t> regionAlignment;

        // How much __TEXT in each subCache before we split to a new file
        CacheVMSize subCacheTextLimit;
    };

    // arm64* layout
    //
    //                                     â”Œâ”€â”€â”€â”€â”€â”€â–¶â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”
    //                                     â”‚       â”‚                            â”‚      â”‚
    // â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”˜       â”‚    __TEXT: Dylib 0 .. n    â”‚      â””â”€â”€â”€â”€â”€â”€â–¶â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
    // â”‚                            â”‚              â”‚                            â”‚              â”‚                            â”‚
    // â”‚     2GB __TEXT+__DATA      â”‚              â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”       â”‚ 110MB __TEXT: Dylib 0 .. k â”‚
    // â”‚                            â”‚              â”‚                            â”‚      â”‚       â”‚                            â”‚
    // â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¼â”€â”€â”€â”€â”€â”€â”       â”‚ __DATA_CONST: Dylib 0 .. n â”‚      â”‚       â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
    // â”‚                            â”‚      â”‚       â”‚                            â”‚      â”‚       â”‚                            â”‚
    // â”‚     2GB __TEXT+__DATA      â”‚      â”‚       â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤      â”‚       â”‚    Stubs: Dylib 0 .. k     â”‚
    // â”‚                            â”‚      â”‚       â”‚                            â”‚      â”‚       â”‚                            â”‚
    // â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤      â”‚       â”‚       Padding: 32MB        â”‚      â”‚       â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
    // â”‚                            â”‚      â”‚       â”‚                            â”‚      â”‚       â”‚                            â”‚
    // â”‚            ...             â”‚      â”‚       â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤      â”‚       â”‚     110MB __TEXT: ...      â”‚
    // â”‚                            â”‚      â”‚       â”‚                            â”‚      â”‚       â”‚                            â”‚
    // â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤      â”‚       â”‚    __DATA: Dylib 0 .. n    â”‚      â”‚       â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
    // â”‚                            â”‚      â”‚       â”‚                            â”‚      â”‚       â”‚                            â”‚
    // â”‚  __TEXT+__DATA+__LINKEDIT  â”‚      â”‚       â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤      â”‚       â”‚         Stubs: ...         â”‚
    // â”‚                            â”‚      â”‚       â”‚                            â”‚      â”‚       â”‚                            â”‚
    // â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜      â”‚       â”‚ __TPRO_CONST: Dylib 0 .. n â”‚      â””â”€â”€â”€â”€â”€â”€â–¶â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    //                                     â”‚       â”‚                            â”‚
    //                                     â”‚       â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
    //                                     â”‚       â”‚                            â”‚
    //                                     â”‚       â”‚    __AUTH: Dylib 0 .. n    â”‚
    //                                     â”‚       â”‚                            â”‚
    //                                     â”‚       â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
    //                                     â”‚       â”‚                            â”‚
    //                                     â”‚       â”‚ __AUTH_CONST: Dylib 0 .. n â”‚
    //                                     â”‚       â”‚                            â”‚
    //                                     â”‚       â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
    //                                     â”‚       â”‚                            â”‚
    //                                     â”‚       â”‚       Padding: 32MB        â”‚
    //                                     â”‚       â”‚                            â”‚
    //                                     â”‚       â”œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¤
    //                                     â”‚       â”‚                            â”‚
    //                                     â”‚       â”‚ __READ_ONLY: Dylib 0 .. n  â”‚
    //                                     â”‚       â”‚                            â”‚
    //                                     â””â”€â”€â”€â”€â”€â”€â–¶â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
    struct Contiguous
    {
        // How many bytes of padding do we add between each Region
        CacheVMSize regionPadding;

        // How much __TEXT + DATA + ... (not LINKEDIT) in each subCache before we split to a new file
        CacheVMSize subCacheTextDataLimit;

        // How much __TEXT before we make a new stubs subCache
        CacheVMSize subCacheStubsLimit;

        // How much padding will we have in every "subCacheTextDataLimit" (2GB) region of memory
        CacheVMSize subCachePadding;
    };

    // Fields for all layouts
    CacheVMAddress          cacheBaseAddress;
    CacheVMSize             cacheSize;
    std::optional<uint64_t> cacheMaxSlide;
    std::optional<uint64_t> cacheFixedSlide;
    const bool              is64;
    const bool              hasAuthRegion;
    const bool              tproIsInData;
    const uint32_t          pageSize;
    const uint32_t          machHeaderAlignment = 4096;

    // Fields only used for discontiguous layouts, ie, x86_64
    std::optional<Discontiguous>    discontiguous;

    // Fields only used for contiguous layouts, ie, arm64*
    std::optional<Contiguous>       contiguous;
};

struct SlideInfo
{
    SlideInfo(const BuilderOptions& options, const Layout& layout);

    enum class SlideInfoFormat
    {
        v1,
        v2,
        v3,
        // v4 (deprecated.  arm64_32 uses v1 instead)
        v5,
    };

    std::optional<SlideInfoFormat>  slideInfoFormat;
    uint32_t                        slideInfoBytesPerDataPage;
    uint32_t                        slideInfoPageSize           = 4096; // 16384 for v5
    CacheVMAddress                  slideInfoValueAdd;
    uint64_t                        slideInfoDeltaMask          = 0;
};

struct CodeSign
{
    CodeSign(const BuilderOptions& options);

    enum class Mode
    {
        onlySHA256,
        onlySHA1,
        agile
    };

    const Mode      mode;
    const uint32_t  pageSize;
};

struct BuilderConfig
{
    BuilderConfig(const BuilderOptions& options);

    Logger      log;
    Timer       timer;
    Layout      layout;
    SlideInfo   slideInfo;
    CodeSign    codeSign;
};

} // namespace cache_builder

#endif /* BuilderConfig_hpp */
