/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 4, 2023.
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
#ifndef _I386_MTRR_H_
#define _I386_MTRR_H_

/*
 * Memory type range register (MTRR) support.
 */

#include <mach/std_types.h>
#include <sys/appleapiopts.h>
#include <sys/cdefs.h>

#ifdef __APPLE_API_PRIVATE

enum {
	MTRR_TYPE_UNCACHEABLE  = 0,
	MTRR_TYPE_WRITECOMBINE = 1,
	MTRR_TYPE_WRITETHROUGH = 4,
	MTRR_TYPE_WRITEPROTECT = 5,
	MTRR_TYPE_WRITEBACK    = 6
};

__BEGIN_DECLS

extern void          mtrr_init(void);
extern kern_return_t mtrr_update_cpu(void);
extern kern_return_t mtrr_update_all_cpus(void);

extern kern_return_t mtrr_range_add(    addr64_t phys_addr,
    uint64_t length,
    uint32_t mem_type);

extern kern_return_t mtrr_range_remove( addr64_t phys_addr,
    uint64_t length,
    uint32_t mem_type);

extern void          pat_init(void);

__END_DECLS

#endif /* __APPLE_API_PRIVATE */

#endif /* !_I386_MTRR_H_ */
