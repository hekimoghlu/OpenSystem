/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 24, 2023.
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

#include "internal.h"
#if defined HAVE_DLADDR
#include <dlfcn.h>
#endif
#if defined HAVE_SYS_PARAM_H
#include <sys/param.h>
#endif
static void* stub_options(int argc, char **argv);
#define ruby_options stub_options
#include <main.c>
#undef ruby_options

void *
stub_options(int argc, char **argv)
{
    char xflag[] = "-x";
    char *xargv[4] = {NULL, xflag};
    char *cmd = argv[0];
    void *ret;

#if defined __CYGWIN__ || defined _WIN32
    /* GetCommandLineW should contain the accessible path,
     * use argv[0] as is */
#elif defined __linux__
    {
	char selfexe[MAXPATHLEN];
	ssize_t len = readlink("/proc/self/exe", selfexe, sizeof(selfexe));
	if (len < 0) {
	    perror("readlink(\"/proc/self/exe\")");
	    return NULL;
	}
	selfexe[len] = '\0';
	cmd = selfexe;
    }
#elif defined HAVE_DLADDR
    {
	Dl_info dli;
	if (!dladdr(stub_options, &dli)) {
	    perror("dladdr");
	    return NULL;
	}
	cmd = (char *)dli.dli_fname;
    }
#endif

#ifndef HAVE_SETPROCTITLE
    /* argc and argv must be the original */
    ruby_init_setproctitle(argc, argv);
#endif

    /* set script with -x option */
    /* xargv[0] is NULL not to re-initialize setproctitle again */
    xargv[2] = cmd;
    ret = ruby_options(3, xargv);

    /* set all arguments to ARGV */
    ruby_set_argv(argc - 1, argv + 1);

    return ret;
}
