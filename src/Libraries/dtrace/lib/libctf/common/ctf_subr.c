/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
 * Copyright 2003 Sun Microsystems, Inc.  All rights reserved.
 * Use is subject to license terms.
 */

#include <ctf_impl.h>
#include <libctf.h>
#include <sys/mman.h>
#include <stdarg.h>

void *
ctf_data_alloc(size_t size)
{
	return (mmap(NULL, size, PROT_READ | PROT_WRITE,
	    MAP_PRIVATE | MAP_ANON, -1, 0));
}

void
ctf_data_free(void *buf, size_t size)
{
	(void) munmap(buf, size);
}

void
ctf_data_protect(void *buf, size_t size)
{
	(void) mprotect(buf, size, PROT_READ);
}

void *
ctf_alloc(size_t size)
{
	return (malloc(size));
}

/*ARGSUSED*/
void
ctf_free(void *buf, size_t size)
{
#pragma unused(size)
	free(buf);
}

const char *
ctf_strerror(int err)
{
	return (strerror(err));
}

/*PRINTFLIKE1*/
__printflike(1, 2)
void
ctf_dprintf(const char *format, ...)
{
	if (_libctf_debug) {
		va_list alist;

		va_start(alist, format);
		(void) fputs("libctf DEBUG: ", stderr);
		(void) vfprintf(stderr, format, alist);
		va_end(alist);
	}
}
