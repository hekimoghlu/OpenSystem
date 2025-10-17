/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 26, 2022.
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
#ifndef AOM_AOM_MEM_INCLUDE_AOM_MEM_INTRNL_H_
#define AOM_AOM_MEM_INCLUDE_AOM_MEM_INTRNL_H_

#include "config/aom_config.h"

#define ADDRESS_STORAGE_SIZE sizeof(size_t)

#ifndef DEFAULT_ALIGNMENT
#if defined(VXWORKS)
/*default addr alignment to use in calls to aom_* functions other than
  aom_memalign*/
#define DEFAULT_ALIGNMENT 32
#else
#define DEFAULT_ALIGNMENT (2 * sizeof(void *)) /* NOLINT */
#endif
#endif

#endif  // AOM_AOM_MEM_INCLUDE_AOM_MEM_INTRNL_H_
