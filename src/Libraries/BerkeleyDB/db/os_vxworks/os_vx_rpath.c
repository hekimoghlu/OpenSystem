/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 19, 2024.
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

#include "iosLib.h"

/*
 * __db_rpath --
 *	Return the last path separator in the path or NULL if none found.
 */
char *
__db_rpath(path)
	const char *path;
{
	const char *s, *last;
	DEV_HDR *dummy;
	char *ptail;

	/*
	 * VxWorks devices can be rooted at any name.  We want to
	 * skip over the device name and not take into account any
	 * PATH_SEPARATOR characters that might be in that name.
	 *
	 * XXX [#2393]
	 * VxWorks supports having a filename directly follow a device
	 * name with no separator.  I.e. to access a file 'xxx' in
	 * the top level directory of a device mounted at "mydrive"
	 * you could say "mydrivexxx" or "mydrive/xxx" or "mydrive\xxx".
	 * We do not support the first usage here.
	 * XXX
	 */
	if ((dummy = iosDevFind((char *)path, &ptail)) == NULL)
		s = path;
	else
		s = ptail;

	last = NULL;
	if (PATH_SEPARATOR[1] != '\0') {
		for (; s[0] != '\0'; ++s)
			if (strchr(PATH_SEPARATOR, s[0]) != NULL)
				last = s;
	} else
		for (; s[0] != '\0'; ++s)
			if (s[0] == PATH_SEPARATOR[0])
				last = s;
	return ((char *)last);
}
