/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 10, 2025.
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
 * AT&T Bell Laboratories
 *
 * 3d fs operations
 * only active for non-shared 3d library
 */

#define mount	______mount

#include <ast.h>

#undef	mount

#include <fs3d.h>

int
fs3d(register int op)
{
	register int	cur;
	register char*	v;
	char		val[sizeof(FS3D_off) + 8];

	static int	fsview;
	static char	on[] = FS3D_on;
	static char	off[] = FS3D_off;

	if (fsview < 0)
		return 0;

	/*
	 * get the current setting
	 */

	if (!fsview && (!getenv("LD_PRELOAD") || mount("", "", 0, NiL)))
		goto nope;
	if (FS3D_op(op) == FS3D_OP_INIT && mount(FS3D_init, NiL, FS3D_VIEW, NiL))
		goto nope;
	if (mount(on, val, FS3D_VIEW|FS3D_GET|FS3D_SIZE(sizeof(val)), NiL))
		goto nope;
	if (v = strchr(val, ' '))
		v++;
	else
		v = val;
	if (!strcmp(v, on))
		cur = FS3D_ON;
	else if (!strncmp(v, off, sizeof(off) - 1) && v[sizeof(off)] == '=')
		cur = FS3D_LIMIT((int)strtol(v + sizeof(off) + 1, NiL, 0));
	else
		cur = FS3D_OFF;
	if (cur != op)
	{
		switch (FS3D_op(op))
		{
		case FS3D_OP_OFF:
			v = off;
			break;
		case FS3D_OP_ON:
			v = on;
			break;
		case FS3D_OP_LIMIT:
			sfsprintf(val, sizeof(val), "%s=%d", off, FS3D_arg(op));
			v = val;
			break;
		default:
			v = 0;
			break;
		}
		if (v && mount(v, NiL, FS3D_VIEW, NiL))
			goto nope;
	}
	fsview = 1;
	return cur;
 nope:
	fsview = -1;
	return 0;
}

/*
 * user code that includes <fs3d.h> will have mount() mapped to fs3d_mount()
 * this restricts the various "standard" mount prototype conflicts to this spot
 * this means that code that includes <fs3d.h> cannot access the real mount
 * (at least without some additional macro hackery
 */

#undef	mount

extern int	mount(const char*, char*, int, void*);

int
fs3d_mount(const char* source, char* target, int flags, void* data)
{
	return mount(source, target, flags, data);
}
