/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 13, 2024.
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
#include "db_config.h"

#include "db_int.h"
#if defined(HAVE_SYSTEM_INCLUDE_FILES) && defined(HAVE_BACKTRACE) && \
    defined(HAVE_BACKTRACE_SYMBOLS) && defined(HAVE_EXECINFO_H)
#include <execinfo.h>
#endif

/*
 * __os_stack --
 *	Output a stack trace to the message file handle.
 *
 * PUBLIC: void __os_stack __P((ENV *));
 */
void
__os_stack(env)
	ENV *env;
{
#if defined(HAVE_BACKTRACE) && defined(HAVE_BACKTRACE_SYMBOLS)
	void *array[200];
	size_t i, size;
	char **strings;

	/*
	 * Solaris and the GNU C library support this interface.  Solaris
	 * has additional interfaces (printstack and walkcontext), I don't
	 * know if they offer any additional value or not.
	 */
	size = backtrace(array, sizeof(array) / sizeof(array[0]));
	strings = backtrace_symbols(array, size);

	for (i = 0; i < size; ++i)
		__db_errx(env, "%s", strings[i]);
	free(strings);
#endif
	COMPQUIET(env, NULL);
}
