/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 14, 2022.
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
/*
 * Written by Marc Espie, September 25, 1999
 * Public domain.
 */

#ifndef	COMPAT_STDBOOL_H
#define	COMPAT_STDBOOL_H

#ifndef __cplusplus

#if (defined(HAVE__BOOL) && HAVE__BOOL > 0) || defined(lint)
/* Support for _C99: type _Bool is already built-in. */
#define false	0
#define true	1

#else
/* `_Bool' type must promote to `int' or `unsigned int'. */
typedef enum {
	false = 0,
	true = 1
} _Bool;

/* And those constants must also be available as macros. */
#define	false	false
#define	true	true

#endif

/* User visible type `bool' is provided as a macro which may be redefined */
#define bool _Bool

#else /* __cplusplus */
#define _Bool 	bool
#define bool 	bool
#define false 	false
#define true 	true
#endif /* __cplusplus */

/* Inform that everything is fine */
#define __bool_true_false_are_defined 1

#endif /* COMPAT_STDBOOL_H */
