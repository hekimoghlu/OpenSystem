/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 12, 2024.
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
#include "newuser.mdh"
#include "newuser.pro"

#include "../zshpaths.h"

/**/
int
setup_(UNUSED(Module m))
{
    return 0;
}

/**/
int
features_(UNUSED(Module m), UNUSED(char ***features))
{
    return 1;
}

/**/
int
enables_(UNUSED(Module m), UNUSED(int **enables))
{
    return 0;
}

/**/
static int
check_dotfile(const char *dotdir, const char *fname)
{
    VARARR(char, buf, strlen(dotdir) + strlen(fname) + 2);
    sprintf(buf, "%s/%s", dotdir, fname);

    return access(buf, F_OK);
}

/**/
int
boot_(UNUSED(Module m))
{
    const char *dotdir = getsparam_u("ZDOTDIR");
    const char *spaths[] = {
#ifdef SITESCRIPT_DIR
	SITESCRIPT_DIR,
#endif
#ifdef SCRIPT_DIR
	SCRIPT_DIR,
#endif
	0 };
    const char **sp;

    if (!EMULATION(EMULATE_ZSH))
	return 0;

    if (!dotdir) {
	dotdir = home;
	if (!dotdir)
	    return 0;
    }

    if (check_dotfile(dotdir, ".zshenv") == 0 ||
	check_dotfile(dotdir, ".zprofile") == 0 ||
	check_dotfile(dotdir, ".zshrc") == 0 ||
	check_dotfile(dotdir, ".zlogin") == 0)
	return 0;

    for (sp = spaths; *sp; sp++) {
	VARARR(char, buf, strlen(*sp) + 9);
	sprintf(buf, "%s/newuser", *sp);

	if (source(buf) != SOURCE_NOT_FOUND)
	    break;
    }

    return 0;
}

/**/
int
cleanup_(UNUSED(Module m))
{
    return 0;
}

/**/
int
finish_(UNUSED(Module m))
{
    return 0;
}
