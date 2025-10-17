/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 29, 2024.
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
#pragma prototyped

#ifndef _NO_LARGEFILE64_SOURCE
#define _NO_LARGEFILE64_SOURCE	1
#endif

#include "stdhdr.h"

long
ftell(Sfio_t* f)
{
	STDIO_INT(f, "ftell", long, (Sfio_t*), (f))

	return (long)sfseek(f, (Sfoff_t)0, SEEK_CUR);
}

#if _typ_int64_t

int64_t
ftell64(Sfio_t* f)
{
	STDIO_INT(f, "ftell64", int64_t, (Sfio_t*), (f))

	return (int64_t)sfseek(f, (Sfoff_t)0, SEEK_CUR);
}

#endif
