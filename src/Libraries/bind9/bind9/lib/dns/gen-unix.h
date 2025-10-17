/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 24, 2023.
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
/* $Id: gen-unix.h,v 1.21 2009/01/17 23:47:42 tbox Exp $ */

/*! \file
 * \brief
 * This file is responsible for defining two operations that are not
 * directly portable between Unix-like systems and Windows NT, option
 * parsing and directory scanning.  It is here because it was decided
 * that the "gen" build utility was not to depend on libisc.a, so
 * the functions declared in isc/commandline.h and isc/dir.h could not
 * be used.
 *
 * The commandline stuff is really just a wrapper around getopt().
 * The dir stuff was shrunk to fit the needs of gen.c.
 */

#ifndef DNS_GEN_UNIX_H
#define DNS_GEN_UNIX_H 1

#include <sys/types.h>          /* Required on some systems for dirent.h. */

#include <dirent.h>
#include <unistd.h>		/* XXXDCL Required for ?. */

#include <isc/boolean.h>
#include <isc/lang.h>

#ifdef NEED_OPTARG
extern char *optarg;
#endif

#define isc_commandline_parse		getopt
#define isc_commandline_argument 	optarg

typedef struct {
	DIR *handle;
	char *filename;
} isc_dir_t;

ISC_LANG_BEGINDECLS

static isc_boolean_t
start_directory(const char *path, isc_dir_t *dir) {
	dir->handle = opendir(path);

	if (dir->handle != NULL)
		return (ISC_TRUE);
	else
		return (ISC_FALSE);

}

static isc_boolean_t
next_file(isc_dir_t *dir) {
	struct dirent *dirent;

	dir->filename = NULL;

	if (dir->handle != NULL) {
		dirent = readdir(dir->handle);
		if (dirent != NULL)
			dir->filename = dirent->d_name;
	}

	if (dir->filename != NULL)
		return (ISC_TRUE);
	else
		return (ISC_FALSE);
}

static void
end_directory(isc_dir_t *dir) {
	if (dir->handle != NULL)
		(void)closedir(dir->handle);

	dir->handle = NULL;
}

ISC_LANG_ENDDECLS

#endif /* DNS_GEN_UNIX_H */
