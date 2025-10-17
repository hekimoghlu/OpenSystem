/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 5, 2025.
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

/*
 * __os_getenv --
 *	Retrieve an environment variable.
 *
 * PUBLIC: int __os_getenv __P((ENV *, const char *, char **, size_t));
 */
int
__os_getenv(env, name, bpp, buflen)
	ENV *env;
	const char *name;
	char **bpp;
	size_t buflen;
{
	/*
	 * If we have getenv, there's a value and the buffer is large enough:
	 *	copy value into the pointer, return 0
	 * If we have getenv, there's a value  and the buffer is too short:
	 *	set pointer to NULL, return EINVAL
	 * If we have getenv and there's no value:
	 *	set pointer to NULL, return 0
	 * If we don't have getenv:
	 *	set pointer to NULL, return 0
	 */
#ifdef HAVE_GETENV
	char *p;

	if ((p = getenv(name)) != NULL) {
		if (strlen(p) < buflen) {
			(void)strcpy(*bpp, p);
			return (0);
		}

		*bpp = NULL;
		__db_errx(env,
		    "%s: buffer too small to hold environment variable %s",
		    name, p);
		return (EINVAL);
	}
#else
	COMPQUIET(env, NULL);
	COMPQUIET(name, NULL);
	COMPQUIET(buflen, 0);
#endif
	*bpp = NULL;
	return (0);
}
