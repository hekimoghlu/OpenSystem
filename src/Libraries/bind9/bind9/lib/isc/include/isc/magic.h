/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 30, 2023.
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
/* $Id: magic.h,v 1.18 2007/06/19 23:47:18 tbox Exp $ */

#ifndef ISC_MAGIC_H
#define ISC_MAGIC_H 1

#include <isc/util.h>

/*! \file isc/magic.h */

typedef struct {
	unsigned int magic;
} isc__magic_t;


/*%
 * To use this macro the magic number MUST be the first thing in the
 * structure, and MUST be of type "unsigned int".
 * The intent of this is to allow magic numbers to be checked even though
 * the object is otherwise opaque.
 */
#define ISC_MAGIC_VALID(a,b)	(ISC_LIKELY((a) != NULL) && \
				 ISC_LIKELY(((const isc__magic_t *)(a))->magic == (b)))

#define ISC_MAGIC(a, b, c, d)	((a) << 24 | (b) << 16 | (c) << 8 | (d))

#endif /* ISC_MAGIC_H */
