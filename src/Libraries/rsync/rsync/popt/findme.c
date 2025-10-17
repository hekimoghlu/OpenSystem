/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 22, 2023.
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
/* (C) 1998-2000 Red Hat, Inc. -- Licensing details are in the COPYING
   file accompanying popt source distributions, available from 
   ftp://ftp.rpm.org/pub/rpm/dist. */

#include "system.h"
#include "findme.h"

const char * findProgramPath(const char * argv0) {
    char * path = getenv("PATH");
    char * pathbuf;
    char * start, * chptr;
    char * buf;

    if (argv0 == NULL) return NULL;	/* XXX can't happen */
    /* If there is a / in the argv[0], it has to be an absolute path */
    if (strchr(argv0, '/'))
	return xstrdup(argv0);

    if (path == NULL) return NULL;

    start = pathbuf = alloca(strlen(path) + 1);
    buf = malloc(strlen(path) + strlen(argv0) + sizeof("/"));
    if (buf == NULL) return NULL;	/* XXX can't happen */
    strcpy(pathbuf, path);

    chptr = NULL;
    /*@-branchstate@*/
    do {
	if ((chptr = strchr(start, ':')))
	    *chptr = '\0';
	sprintf(buf, "%s/%s", start, argv0);

	if (!access(buf, X_OK))
	    return buf;

	if (chptr) 
	    start = chptr + 1;
	else
	    start = NULL;
    } while (start && *start);
    /*@=branchstate@*/

    free(buf);

    return NULL;
}
