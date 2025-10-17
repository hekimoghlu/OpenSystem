/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 2, 2024.
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

/*
 * C99 stdio extensions
 */

#include "stdhdr.h"

void
clearerr_unlocked(Sfio_t* sp)
{
	clearerr(sp);
}

int
feof_unlocked(Sfio_t* sp)
{
	return feof(sp);
}

int
ferror_unlocked(Sfio_t* sp)
{
	return ferror(sp);
}

int
fflush_unlocked(Sfio_t* sp)
{
	return fflush(sp);
}

int
fgetc_unlocked(Sfio_t* sp)
{
	return fgetc(sp);
}

char*
fgets_unlocked(char* buf, int size, Sfio_t* sp)
{
	return fgets(buf, size, sp);
}

int
fileno_unlocked(Sfio_t* sp)
{
	return fileno(sp);
}

int
fputc_unlocked(int c, Sfio_t* sp)
{
	return fputc(c, sp);
}

int
fputs_unlocked(char* buf, Sfio_t* sp)
{
	return fputs(buf, sp);
}

size_t
fread_unlocked(void* buf, size_t size, size_t n, Sfio_t* sp)
{
	return fread(buf, size, n, sp);
}

size_t
fwrite_unlocked(void* buf, size_t size, size_t n, Sfio_t* sp)
{
	return fwrite(buf, size, n, sp);
}

int
getc_unlocked(Sfio_t* sp)
{
	return getc(sp);
}

int
getchar_unlocked(void)
{
	return getchar();
}

int
putc_unlocked(int c, Sfio_t* sp)
{
	return putc(c, sp);
}

int
putchar_unlocked(int c)
{
	return putchar(c);
}
