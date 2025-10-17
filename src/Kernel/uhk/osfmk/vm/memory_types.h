/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 16, 2024.
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
/* machine independent WIMG bits */

#ifndef _VM_MEMORY_TYPES_H_
#define _VM_MEMORY_TYPES_H_

#include <machine/memory_types.h>

#define VM_MEM_GUARDED          0x1             /* (G) Guarded Storage */
#define VM_MEM_COHERENT         0x2             /* (M) Memory Coherency */
#define VM_MEM_NOT_CACHEABLE    0x4             /* (I) Cache Inhibit */
#define VM_MEM_WRITE_THROUGH    0x8             /* (W) Write-Through */

#define VM_WIMG_USE_DEFAULT     0x80
#define VM_WIMG_MASK            0xFF

#define HAS_DEFAULT_CACHEABILITY(attr)                                  \
	                        (                                       \
	                        ((attr) == VM_WIMG_USE_DEFAULT) ||      \
	                        ((attr) == VM_WIMG_DEFAULT)             \
	                        )

#endif /* _VM_MEMORY_TYPES_H_ */
