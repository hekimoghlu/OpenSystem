/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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
#ifndef debuggerSuppot_h
#define debuggerSuppot_h

#include <mach-o/dyld_images.h>

#include <span>

#include "Defines.h"

namespace lsl {
    struct Allocator;
}
namespace dyld4 {
    class RuntimeState;
    void addImagesToAllImages(lsl::Allocator& persistentAllocator, uint32_t infoCount, const dyld_image_info info[],  uint32_t initialImageCount);
    void removeImageFromAllImages(const mach_header* loadAddress);
    void addNonSharedCacheImageUUID(lsl::Allocator&, const dyld_uuid_info& info);
#if TARGET_OS_SIMULATOR
    void syncProcessInfo(Allocator& allocator);
#endif

#if SUPPORT_ROSETTA
    void addAotImagesToAllAotImages(lsl::Allocator&, uint32_t aotInfoCount, const dyld_aot_image_info aotInfo[]);
    void removeAotImageFromAllAotImages(const mach_header* loadAddress);
#endif
}

extern "C" void lldb_image_notifier(enum dyld_image_mode mode, uint32_t infoCount, const dyld_image_info info[]);

extern dyld_all_image_infos*        gProcessInfo;

#endif /* debuggerSuppot_h */
