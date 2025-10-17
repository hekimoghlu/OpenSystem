/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 17, 2025.
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
#include <sys/cdefs.h>
__FBSDID("$FreeBSD: src/lib/libc/locale/collcmp.c,v 1.18 2005/02/27 14:54:23 phantom Exp $");

#include <xlocale.h>
#include <wchar.h>
#include "collate.h"

/*
 * Compare two characters using collate
 */

__private_extern__ int
__collate_range_cmp(wchar_t c1, wchar_t c2, locale_t loc)
{
	static wchar_t s1[2], s2[2];

	s1[0] = c1;
	s2[0] = c2;
	return (wcscoll_l(s1, s2, loc));
}
