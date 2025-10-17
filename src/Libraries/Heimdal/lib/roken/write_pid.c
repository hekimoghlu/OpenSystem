/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 18, 2023.
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
#include <config.h>

#include "roken.h"

ROKEN_LIB_FUNCTION char * ROKEN_LIB_CALL
pid_file_write (const char *progname)
{
    char *ret = NULL;
    FILE *fp;

    if (asprintf (&ret, "%s%s.pid", _PATH_VARRUN, progname) < 0 || ret == NULL)
	return NULL;
    fp = fopen (ret, "w");
    if (fp == NULL) {
	free (ret);
	return NULL;
    }
    fprintf (fp, "%u", (unsigned)getpid());
    fclose (fp);
    return ret;
}

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
pid_file_delete (char **filename)
{
    if (*filename != NULL) {
	unlink (*filename);
	free (*filename);
	*filename = NULL;
    }
}

#ifndef HAVE_PIDFILE
static char *pidfile_path;

static void
pidfile_cleanup(void)
{
    if(pidfile_path != NULL)
	pid_file_delete(&pidfile_path);
}

ROKEN_LIB_FUNCTION void ROKEN_LIB_CALL
pidfile(const char *basename)
{
    if(pidfile_path != NULL)
	return;
    if(basename == NULL)
	basename = getprogname();
    pidfile_path = pid_file_write(basename);
#if defined(HAVE_ATEXIT)
    atexit(pidfile_cleanup);
#elif defined(HAVE_ON_EXIT)
    on_exit(pidfile_cleanup);
#endif
}
#endif
