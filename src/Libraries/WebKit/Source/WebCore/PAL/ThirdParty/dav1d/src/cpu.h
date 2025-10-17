/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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
#ifndef DAV1D_SRC_CPU_H
#define DAV1D_SRC_CPU_H

#include "config.h"

#include "common/attributes.h"

#include "dav1d/common.h"
#include "dav1d/dav1d.h"

#if ARCH_AARCH64 || ARCH_ARM
#include "src/arm/cpu.h"
#elif ARCH_PPC64LE
#include "src/ppc/cpu.h"
#elif ARCH_X86
#include "src/x86/cpu.h"
#endif

void dav1d_init_cpu(void);
unsigned dav1d_get_cpu_flags(void);
DAV1D_API void dav1d_set_cpu_flags_mask(unsigned mask);
int dav1d_num_logical_processors(Dav1dContext *c);

#endif /* DAV1D_SRC_CPU_H */
