/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 15, 2023.
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
/* $Id$ */

#ifndef ISC_SAFE_H
#define ISC_SAFE_H 1

/*! \file isc/safe.h */

#include <isc/types.h>

ISC_LANG_BEGINDECLS

isc_boolean_t
isc_safe_memequal(const void *s1, const void *s2, size_t n);
/*%<
 * Returns ISC_TRUE iff. two blocks of memory are equal, otherwise
 * ISC_FALSE.
 *
 */

int
isc_safe_memcompare(const void *b1, const void *b2, size_t len);
/*%<
 * Clone of libc memcmp() which is safe to differential timing attacks.
 */

ISC_LANG_ENDDECLS

#endif /* ISC_SAFE_H */
