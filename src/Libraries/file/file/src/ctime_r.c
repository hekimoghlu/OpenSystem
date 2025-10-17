/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 9, 2022.
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
#include "file.h"
#ifndef	lint
FILE_RCSID("@(#)$File: ctime_r.c,v 1.1 2012/05/15 17:14:36 christos Exp $")
#endif	/* lint */
#include <time.h>
#include <string.h>

/* ctime_r is not thread-safe anyway */
char *
ctime_r(const time_t *t, char *dst)
{
	char *p = ctime(t);
	if (p == NULL)
		return NULL;
	memcpy(dst, p, 26);
	return dst;
}
