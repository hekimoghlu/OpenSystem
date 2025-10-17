/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 10, 2024.
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
FILE_RCSID("@(#)$File: localtime_r.c,v 1.2 2015/07/11 14:41:37 christos Exp $")
#endif	/* lint */
#include <time.h>
#include <string.h>

/* asctime_r is not thread-safe anyway */
struct tm *
localtime_r(const time_t *t, struct tm *tm)
{
	struct tm *tmp = localtime(t);
	if (tmp == NULL)
		return NULL;
	memcpy(tm, tmp, sizeof(*tm));
	return tmp;
}
