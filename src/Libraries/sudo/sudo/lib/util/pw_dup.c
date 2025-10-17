/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 18, 2025.
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
 * This is an open source non-commercial project. Dear PVS-Studio, please check it.
 * PVS-Studio Static Code Analyzer for C, C++ and C#: http://www.viva64.com
 */

#include <config.h>

#ifndef HAVE_PW_DUP

#include <stdlib.h>
#include <string.h>
#include <pwd.h>

#include "sudo_compat.h"

#define PW_SIZE(name, size)				\
do {							\
	if (pw->name) {					\
		size = strlen(pw->name) + 1;		\
		total += size;				\
	}						\
} while (0)

#define PW_COPY(name, size)				\
do {							\
	if (pw->name) {					\
		(void)memcpy(cp, pw->name, size);	\
		newpw->name = cp;			\
		cp += size;				\
	}						\
} while (0)

struct passwd *
sudo_pw_dup(const struct passwd *pw)
{
	size_t nsize = 0, psize = 0, gsize = 0, dsize = 0, ssize = 0, total;
#ifdef HAVE_LOGIN_CAP_H
	size_t csize = 0;
#endif
	struct passwd *newpw;
	char *cp;

	/* Allocate in one big chunk for easy freeing */
	total = sizeof(struct passwd);
	PW_SIZE(pw_name, nsize);
	PW_SIZE(pw_passwd, psize);
#ifdef HAVE_LOGIN_CAP_H
	PW_SIZE(pw_class, csize);
#endif
	PW_SIZE(pw_gecos, gsize);
	PW_SIZE(pw_dir, dsize);
	PW_SIZE(pw_shell, ssize);

	if ((cp = malloc(total)) == NULL)
		return NULL;
	newpw = (struct passwd *)cp;

	/*
	 * Copy in passwd contents and make strings relative to space
	 * at the end of the buffer.
	 */
	(void)memcpy(newpw, pw, sizeof(struct passwd));
	cp += sizeof(struct passwd);

	PW_COPY(pw_name, nsize);
	PW_COPY(pw_passwd, psize);
#ifdef HAVE_LOGIN_CAP_H
	PW_COPY(pw_class, csize);
#endif
	PW_COPY(pw_gecos, gsize);
	PW_COPY(pw_dir, dsize);
	PW_COPY(pw_shell, ssize);

	return newpw;
}
#endif /* HAVE_PW_DUP */
