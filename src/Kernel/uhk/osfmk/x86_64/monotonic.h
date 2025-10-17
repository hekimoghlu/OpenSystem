/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 17, 2024.
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
#ifndef X86_64_MONOTONIC_H
#define X86_64_MONOTONIC_H

#include <stdbool.h>
#include <stdint.h>

#define MT_NDEVS 1

#define MT_CORE_NFIXED 4

#define MT_CORE_INSTRS 0
#define MT_CORE_CYCLES 1
#define MT_CORE_REFCYCLES 2
#define MT_CORE_MAXVAL ((UINT64_C(1) << 48) - 1)

extern bool mt_core_supported;

#endif /* !defined(X86_64_MONOTONIC_H) */
