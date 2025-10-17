/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 10, 2022.
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
#ifndef _LIBC_THREAD_STACK_PCS_H
#define _LIBC_THREAD_STACK_PCS_H

#include <_bounds.h>
#include <sys/cdefs.h>
#include <mach/vm_statistics.h>
#include <mach/vm_types.h>

_LIBC_SINGLE_BY_DEFAULT()

__BEGIN_DECLS

extern int thread_stack_pcs(vm_address_t *_LIBC_COUNT(max) buffer, unsigned max, unsigned *num);
extern int thread_stack_async_pcs(vm_address_t *_LIBC_COUNT(max) buffer, unsigned max, unsigned *num);

__END_DECLS

#endif // _LIBC_THREAD_STACK_PCS_H
