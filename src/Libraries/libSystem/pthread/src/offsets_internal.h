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
#ifndef _POSIX_PTHREAD_OFFSETS_H
#define _POSIX_PTHREAD_OFFSETS_H

#if defined(__i386__)
#define _PTHREAD_STRUCT_DIRECT_STACKADDR_OFFSET   140
#define _PTHREAD_STRUCT_DIRECT_STACKBOTTOM_OFFSET 144
#elif __LP64__
#define _PTHREAD_STRUCT_DIRECT_STACKADDR_OFFSET   -48
#define _PTHREAD_STRUCT_DIRECT_STACKBOTTOM_OFFSET -40
#else
#define _PTHREAD_STRUCT_DIRECT_STACKADDR_OFFSET   -36
#define _PTHREAD_STRUCT_DIRECT_STACKBOTTOM_OFFSET -32
#endif

#ifndef __ASSEMBLER__
#include "pthread/private.h" // for other _PTHREAD_STRUCT_DIRECT_*_OFFSET

#define check_backward_offset(field, value) \
		_Static_assert(offsetof(struct pthread_s, tsd) + value == \
				offsetof(struct pthread_s, field), #value " is correct")
#define check_forward_offset(field, value) \
		_Static_assert(offsetof(struct pthread_s, field) == value, \
				#value " is correct")

check_forward_offset(tsd, _PTHREAD_STRUCT_DIRECT_TSD_OFFSET);
check_backward_offset(thread_id, _PTHREAD_STRUCT_DIRECT_THREADID_OFFSET);
#if defined(__i386__)
check_forward_offset(stackaddr, _PTHREAD_STRUCT_DIRECT_STACKADDR_OFFSET);
check_forward_offset(stackbottom, _PTHREAD_STRUCT_DIRECT_STACKBOTTOM_OFFSET);
#else
check_backward_offset(stackaddr, _PTHREAD_STRUCT_DIRECT_STACKADDR_OFFSET);
check_backward_offset(stackbottom, _PTHREAD_STRUCT_DIRECT_STACKBOTTOM_OFFSET);
#endif

#endif // __ASSEMBLER__

#endif /* _POSIX_PTHREAD_OFFSETS_H */
