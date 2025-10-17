/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 1, 2021.
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
#ifndef _NTFS_SUPPORT_H
#define _NTFS_SUPPORT_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_STDDEF_H
#include <stddef.h>
#endif

/*
 * Our mailing list. Use this define to prevent typos in email address.
 */
#define NTFS_DEV_LIST	"ntfs-support@tuxera.com"

/*
 * Generic macro to convert pointers to values for comparison purposes.
 */
#ifndef p2n
#define p2n(p)		((ptrdiff_t)((ptrdiff_t*)(p)))
#endif

/*
 * The classic min and max macros.
 */
#ifndef min
#define min(a,b)	((a) <= (b) ? (a) : (b))
#endif

#ifndef max
#define max(a,b)	((a) >= (b) ? (a) : (b))
#endif

/*
 * Useful macro for determining the offset of a struct member.
 */
#ifndef offsetof
#define offsetof(TYPE, MEMBER) ((size_t) &((TYPE *)0)->MEMBER)
#endif

/*
 * Simple bit operation macros. NOTE: These are NOT atomic.
 */
#define test_bit(bit, var)	      ((var) & (1 << (bit)))
#define set_bit(bit, var)	      (var) |= 1 << (bit)
#define clear_bit(bit, var)	      (var) &= ~(1 << (bit))

#define test_and_set_bit(bit, var)			\
({							\
	const BOOL old_state = test_bit(bit, var);	\
	set_bit(bit, var);				\
	old_state;					\
})

#define test_and_clear_bit(bit, var)			\
({							\
	const BOOL old_state = test_bit(bit, var);	\
	clear_bit(bit, var);				\
	old_state;					\
})

#endif /* defined _NTFS_SUPPORT_H */
