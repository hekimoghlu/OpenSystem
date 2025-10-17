/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 18, 2021.
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
#ifndef BuilderOptions_hpp
#define BuilderOptions_hpp

#include "Error.h"
#include "JSON.h"
#include "MachOFile.h"
#include "Platform.h"

#include <string>
#include <unordered_map>

namespace cache_builder
{

enum class LocalSymbolsMode {
    keep,
    unmap,
    strip
};

enum class CacheKind
{
    development,
    universal
};

struct BuilderOptions
{
    BuilderOptions(std::string_view archName, mach_o::Platform platform,
                   bool dylibsRemovedFromDisk, bool isLocallyBuiltCache,
                   CacheKind kind, bool forceDevelopmentSubCacheSuffix);

    bool isSimulator() const;
    bool isExclaveKit() const;

    // Core fields
    const dyld3::GradedArchs&                   archs;
    std::string                                 mainCacheFileName;
    mach_o::Platform                            platform;
    bool                                        dylibsRemovedFromDisk;
    bool                                        isLocallyBuiltCache;
    bool                                        forceDevelopmentSubCacheSuffix;
    CacheKind                                   kind;
    LocalSymbolsMode                            localSymbolsMode;

    // Logging/printing
    std::string                                 logPrefix;
    bool                                        timePasses   = false;
    bool                                        stats        = false;
    bool                                        debug        = false;

    // Other
    std::unordered_map<std::string, unsigned>   dylibOrdering;
    std::unordered_map<std::string, unsigned>   dirtyDataSegmentOrdering;
    json::Node                                  objcOptimizations;
    std::string                                 swiftGenericMetadataFile;
    std::string                                 prewarmingOptimizations;
};

// Inputs to the builder can be dylibs, executables, bundles, etc.
struct InputFile
{
    const dyld3::MachOFile* mf      = nullptr;
    uint64_t                inode   = 0;
    uint64_t                mtime   = 0;
    uint64_t                size    = 0;
    std::string             path;
    bool                    forceNotCacheEligible = false;

    bool                            hasError() const;
    std::span<const error::Error>   getErrors() const;
    void                            addError(error::Error&& err);

private:
    // These are reason(s) this input file can't be used.  If its a dylib then
    // its likely reasons the dylib is ineligible to be in the cache.  If its
    // an executable then likely the reason the executable can't get a prebuilt closure.
    // Despite the name, these may not be surfaced as errors to the user, as these
    // might not be fatal errors, but at the very least they'll be warnings the user
    // can choose to be fatal or not
    std::vector<error::Error> errors;
};

// Alias for an input file
struct FileAlias
{
    std::string             realPath;
    std::string             aliasPath;
};

} // namespace cache_builder

#endif /* BuilderOptions_hpp */
