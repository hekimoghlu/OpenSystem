/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 28, 2024.
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
#ifndef _DYLD_SHARED_CACHE_EXTRACTOR_
#define _DYLD_SHARED_CACHE_EXTRACTOR_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif 

extern int dyld_shared_cache_extract_dylibs(const char* shared_cache_file_path, const char* extraction_root_path);
extern int dyld_shared_cache_extract_dylibs_progress(const char* shared_cache_file_path, const char* extraction_root_path,
													void (^progress)(unsigned current, unsigned total));

#ifdef __cplusplus
}
#endif 

#endif // _DYLD_SHARED_CACHE_EXTRACTOR_

