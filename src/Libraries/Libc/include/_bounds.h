/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 3, 2022.
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
#ifndef _LIBC_BOUNDS_H_
#define _LIBC_BOUNDS_H_

#include <sys/cdefs.h>

#ifdef __LIBC_STAGED_BOUNDS_SAFETY_ATTRIBUTES /* compiler-defined */

#define _LIBC_COUNT(x)		__counted_by(x)
#define _LIBC_COUNT_OR_NULL(x)	__counted_by_or_null(x)
#define _LIBC_SIZE(x)		__sized_by(x)
#define _LIBC_SIZE_OR_NULL(x)	__sized_by_or_null(x)
#define _LIBC_ENDED_BY(x)	__ended_by(x)
#define _LIBC_SINGLE		__single
#define _LIBC_UNSAFE_INDEXABLE	__unsafe_indexable
#define _LIBC_CSTR		__null_terminated
#define _LIBC_NULL_TERMINATED   __null_terminated
#define _LIBC_FLEX_COUNT(FIELD, INTCOUNT)	__counted_by(FIELD)

#define _LIBC_SINGLE_BY_DEFAULT()	__ptrcheck_abi_assume_single()
#define _LIBC_PTRCHECK_REPLACED(R)  __ptrcheck_unavailable_r(R)

#else /* _LIBC_ANNOTATE_BOUNDS */

#define _LIBC_COUNT(x)
#define _LIBC_COUNT_OR_NULL(x)
#define _LIBC_SIZE(x)
#define _LIBC_SIZE_OR_NULL(x)
#define _LIBC_ENDED_BY(x)
#define _LIBC_SINGLE
#define _LIBC_UNSAFE_INDEXABLE
#define _LIBC_CSTR
#define _LIBC_NULL_TERMINATED
#define _LIBC_FLEX_COUNT(FIELD, INTCOUNT)	(INTCOUNT)

#define _LIBC_SINGLE_BY_DEFAULT()
#define _LIBC_PTRCHECK_REPLACED(R)

#endif /* _LIBC_ANNOTATE_BOUNDS */

#endif /* _LIBC_BOUNDS_H_ */
