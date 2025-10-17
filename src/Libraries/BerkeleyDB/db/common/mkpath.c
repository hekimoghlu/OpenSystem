/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 29, 2023.
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
 * __db_mkpath -- --
 *	Create intermediate directories.
 *
 * PUBLIC: int __db_mkpath __P((ENV *, const char *));
 */
int
__db_mkpath(env, name)
	ENV *env;
	const char *name;
{
	size_t len;
	int ret;
	char *p, *t, savech;

	/*
	 * Get a copy so we can modify the string.  It's a path and potentially
	 * quite long, so don't allocate the space on the stack.
	 */
	len = strlen(name) + 1;
	if ((ret = __os_malloc(env, len, &t)) != 0)
		return (ret);
	memcpy(t, name, len);

	/*
	 * Cycle through the path, creating intermediate directories.
	 *
	 * Skip the first byte if it's a path separator, it's the start of an
	 * absolute pathname.
	 */
	if (PATH_SEPARATOR[1] == '\0') {
		for (p = t + 1; p[0] != '\0'; ++p)
			if (p[0] == PATH_SEPARATOR[0]) {
				savech = *p;
				*p = '\0';
				if (__os_exists(env, t, NULL) &&
				    (ret = __os_mkdir(
					env, t, env->dir_mode)) != 0)
					break;
				*p = savech;
			}
	} else
		for (p = t + 1; p[0] != '\0'; ++p)
			if (strchr(PATH_SEPARATOR, p[0]) != NULL) {
				savech = *p;
				*p = '\0';
				if (__os_exists(env, t, NULL) &&
				    (ret = __os_mkdir(
					env, t, env->dir_mode)) != 0)
					break;
				*p = savech;
			}

	__os_free(env, t);
	return (ret);
}
