/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 2, 2022.
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
#ifndef other_tools_MiscFileUtils_h
#define other_tools_MiscFileUtils_h

// mach_o
#include "Header.h"

#include "DyldSharedCache.h"

#include <span>

// FIXME: Maybe this should be something like tools_common?
namespace other_tools {

bool withReadOnlyMappedFile(const char* path, void (^handler)(std::span<const uint8_t>)) VIS_HIDDEN;

// used by command line tools to process files that may be on disk or in dyld cache
void forSelectedSliceInPaths(std::span<const char*> paths, std::span<const char*> archFilter, const DyldSharedCache* dyldCache,
                             void (^callback)(const char* slicePath, const mach_o::Header* sliceHeader, size_t sliceLength)) VIS_HIDDEN;

void forSelectedSliceInPaths(std::span<const char*> paths, std::span<const char*> archFilter,
                             void (^callback)(const char* slicePath, const mach_o::Header* sliceHeader, size_t sliceLength)) VIS_HIDDEN;

} // namespace other_tools

#endif // other_tools_MiscFileUtils_h
