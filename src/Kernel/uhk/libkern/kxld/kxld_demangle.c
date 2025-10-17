/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 18, 2025.
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
#if !KERNEL

#include <stdlib.h>

/* This demangler is part of the C++ ABI.  We don't include it directly from
 * <cxxabi.h> so that we can avoid using C++ in the kernel linker.
 */
extern char *
__cxa_demangle(const char* __mangled_name, char* __output_buffer,
    size_t* __length, int* __status);

#endif /* !KERNEL */

#include "kxld_demangle.h"

/*******************************************************************************
*******************************************************************************/
const char *
kxld_demangle(const char *str, char **buffer __unused, size_t *length __unused)
{
#if KERNEL
	return str;
#else
	const char *rval = NULL;
	char *demangled = NULL;
	int status;

	rval = str;

	if (!buffer || !length) {
		goto finish;
	}

	/* Symbol names in the symbol table have an extra '_' prepended to them,
	 * so we skip the first character to make the demangler happy.
	 */
	demangled = __cxa_demangle(str + 1, *buffer, length, &status);
	if (!demangled || status) {
		goto finish;
	}

	*buffer = demangled;
	rval = demangled;
finish:
	return rval;
#endif
}
