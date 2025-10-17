/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 4, 2022.
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
 * coshell procrun(3)
 */

#include "colib.h"

#include <proc.h>

int
coprocrun(const char* path, char** argv, int flags)
{
	register char*		s;
	register char**		a;
	register Sfio_t*	tmp;
	int			n;

	if (!(a = argv))
		return procclose(procopen(path, a, NiL, NiL, PROC_FOREGROUND|PROC_GID|PROC_UID|flags));
	if (!(tmp = sfstropen()))
		return -1;
	sfputr(tmp, path ? path : "sh", -1);
	while (s = *++a)
	{
		sfputr(tmp, " '", -1);
		coquote(tmp, s, 0);
		sfputc(tmp, '\'');
	}
	if (!(s = costash(tmp)))
		return -1;
	n = cosystem(s);
	sfstrclose(tmp);
	return n;
}
