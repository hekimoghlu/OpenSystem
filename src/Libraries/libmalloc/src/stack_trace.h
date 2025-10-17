/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 2, 2025.
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
#ifndef _STACK_TRACE_H_
#define _STACK_TRACE_H_

#include "base.h"

#include <mach/vm_types.h>
#include <stddef.h>

MALLOC_NOEXPORT MALLOC_NOINLINE
size_t
trace_collect(uint8_t *buffer, size_t size);

MALLOC_NOEXPORT
uint32_t
trace_decode(const uint8_t *buffer, size_t size, vm_address_t *addrs, uint32_t num_addrs);

#endif // _STACK_TRACE_H_
