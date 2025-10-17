/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 14, 2024.
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
#ifndef VPX_VPX_MEM_INCLUDE_VPX_MEM_INTRNL_H_
#define VPX_VPX_MEM_INCLUDE_VPX_MEM_INTRNL_H_
#include "./vpx_config.h"

#define ADDRESS_STORAGE_SIZE sizeof(size_t)

#ifndef DEFAULT_ALIGNMENT
#if defined(VXWORKS)
/*default addr alignment to use in calls to vpx_* functions other than
 * vpx_memalign*/
#define DEFAULT_ALIGNMENT 32
#else
#define DEFAULT_ALIGNMENT (2 * sizeof(void *)) /* NOLINT */
#endif
#endif

/*returns an addr aligned to the byte boundary specified by align*/
#define align_addr(addr, align) \
  (void *)(((size_t)(addr) + ((align)-1)) & ~(size_t)((align)-1))

#endif  // VPX_VPX_MEM_INCLUDE_VPX_MEM_INTRNL_H_
