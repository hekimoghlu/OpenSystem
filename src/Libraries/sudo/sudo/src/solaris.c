/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 23, 2024.
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

#ifdef HAVE_PROJECT_H
# include <project.h>
# include <sys/task.h>
# include <errno.h>
# include <pwd.h>
#endif

#include "sudo.h"
#include "sudo_dso.h"

int
os_init(int argc, char *argv[], char *envp[])
{
    /*
     * Solaris 11 is unable to load the per-locale shared objects
     * without this.  We must keep the handle open for it to work.
     * This bug was fixed in Solaris 11 Update 1.
     */
    void *handle = sudo_dso_load("/usr/lib/locale/common/methods_unicode.so.3",
	SUDO_DSO_LAZY|SUDO_DSO_GLOBAL);
    (void)&handle;

    return os_init_common(argc, argv, envp);
}

#ifdef HAVE_PROJECT_H
void
set_project(struct passwd *pw)
{
    struct project proj;
    char buf[PROJECT_BUFSZ];
    int errval;
    debug_decl(set_project, SUDO_DEBUG_UTIL);

    /*
     * Collect the default project for the user and settaskid
     */
    setprojent();
    if (getdefaultproj(pw->pw_name, &proj, buf, sizeof(buf)) != NULL) {
	errval = setproject(proj.pj_name, pw->pw_name, TASK_NORMAL);
	switch(errval) {
	case 0:
	    break;
	case SETPROJ_ERR_TASK:
	    switch (errno) {
	    case EAGAIN:
		sudo_warnx("%s", U_("resource control limit has been reached"));
		break;
	    case ESRCH:
		sudo_warnx(U_("user \"%s\" is not a member of project \"%s\""),
		    pw->pw_name, proj.pj_name);
		break;
	    case EACCES:
		sudo_warnx("%s", U_("the invoking task is final"));
		break;
	    default:
		sudo_warnx(U_("could not join project \"%s\""), proj.pj_name);
		break;
	    }
	    break;
	case SETPROJ_ERR_POOL:
	    switch (errno) {
	    case EACCES:
		sudo_warnx(U_("no resource pool accepting default bindings "
		    "exists for project \"%s\""), proj.pj_name);
		break;
	    case ESRCH:
		sudo_warnx(U_("specified resource pool does not exist for "
		    "project \"%s\""), proj.pj_name);
		break;
	    default:
		sudo_warnx(U_("could not bind to default resource pool for "
		    "project \"%s\""), proj.pj_name);
		break;
	    }
	    break;
	default:
	    if (errval <= 0) {
		sudo_warnx(U_("setproject failed for project \"%s\""), proj.pj_name);
	    } else {
		sudo_warnx(U_("warning, resource control assignment failed for "
		    "project \"%s\""), proj.pj_name);
	    }
	    break;
	}
    } else {
	sudo_warn("getdefaultproj");
    }
    endprojent();
    debug_return;
}
#endif /* HAVE_PROJECT_H */
