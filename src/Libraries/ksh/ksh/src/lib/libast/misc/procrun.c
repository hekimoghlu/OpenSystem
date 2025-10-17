/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 4, 2025.
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
 * Glenn Fowler
 * AT&T Research
 *
 * procopen() + procclose()
 * no env changes
 * no modifications
 * effective=real
 * parent ignores INT & QUIT
 */

#include "proclib.h"

int
procrun(const char* path, char** argv, int flags)
{
#if __OBSOLETE__ < 20090101
	flags &= argv ? PROC_ARGMOD : PROC_CHECK;
#endif
	if (flags & PROC_CHECK)
	{
		char	buf[PATH_MAX];

		return pathpath(path, NiL, PATH_REGULAR|PATH_EXECUTE, buf, sizeof(buf)) ? 0 : -1;
	}
	return procclose(procopen(path, argv, NiL, NiL, flags|PROC_FOREGROUND|PROC_GID|PROC_UID));
}
