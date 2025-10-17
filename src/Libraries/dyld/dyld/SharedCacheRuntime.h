/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 22, 2023.
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
#ifndef __DYLD_SHARED_CACHE_RUNTIME_H__
#define __DYLD_SHARED_CACHE_RUNTIME_H__

#include <string.h>
#include <stdint.h>

#include "Allocator.h"
#include "DyldSharedCache.h"
#include "Platform.h"

namespace dyld4 {
    class ProcessConfig;
}

using dyld4::ProcessConfig;

namespace dyld3 {

struct SharedCacheOptions {
#if !TARGET_OS_EXCLAVEKIT
    // Note we look in the default cache dir first, and then the fallback, if not null
    int                 cacheDirFD = -1;
#else
    DyldSharedCache*     cacheHeader = nullptr;
    size_t               cacheSize;
    const char*          cachePath   = nullptr;
#endif // !TARGET_OS_EXCLAVEKIT
    bool                 forcePrivate;
    bool                 useHaswell;
    bool                 verbose;
    bool                 disableASLR;
    bool                 enableReadOnlyDataConst;
    bool                 enableTPRO;
    bool                 preferCustomerCache;
    bool                 forceDevCache;
    bool                 isTranslated;
    bool                 usePageInLinking;
    mach_o::Platform     platform;
    const ProcessConfig* config = nullptr;
};

struct SharedCacheLoadInfo {
    const DyldSharedCache*      loadAddress     = nullptr;
    long                        slide           = 0;
    const char*                 errorMessage    = nullptr;
    bool                        cacheFileFound  = false;
    bool                        development     = false;
#if !TARGET_OS_EXCLAVEKIT
    FileIdTuple                 cacheFileID;
#endif // !TARGET_OS_EXCLAVEKIT
};

bool loadDyldCache(const SharedCacheOptions& options, SharedCacheLoadInfo* results);

struct SharedCacheFindDylibResults {
    const mach_header*          mhInCache;
    const char*                 pathInCache;
    long                        slideInCache;
};

bool findInSharedCacheImage(const SharedCacheLoadInfo& loadInfo, const char* dylibPathToFind, SharedCacheFindDylibResults* results);

bool pathIsInSharedCacheImage(const SharedCacheLoadInfo& loadInfo, const char* dylibPathToFind);

void deallocateExistingSharedCache();


} // namespace dyld3

#endif // __DYLD_SHARED_CACHE_RUNTIME_H__


