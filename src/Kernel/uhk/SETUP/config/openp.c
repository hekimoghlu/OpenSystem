/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 5, 2022.
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
/*  openp, fopenp  --  search pathlist and open file
 *
 *  Usage:
 *	i = openp (path,file,complete,flags,mode)
 *	f = fopenp (path,file,complete,type)
 *	int i,flags,mode;
 *	FILE *f;
 *	char *path,*file,*complete,*type;
 *
 *  Openp searches for "file" in the pathlist "path";
 *  when the file is found and can be opened by open()
 *  with the specified "flags" and "mode", then the full filename
 *  is copied into "complete" and openp returns the file
 *  descriptor.  If no such file is found, openp returns -1.
 *  Fopenp performs the same function, using fopen() instead
 *  of open() and type instead of flags/mode; it returns 0 if no
 *  file is found.
 *
 *  HISTORY
 * 30-Apr-85  Steven Shafer (sas) at Carnegie-Mellon University
 *	Adapted for 4.2 BSD UNIX.  Added new parameter to openp.c;
 *	changed names of flags, mode, and type parameters to reflect
 *	current manual entries for open and fopen.
 *
 * 20-Nov-79  Steven Shafer (sas) at Carnegie-Mellon University
 *	Created for VAX.
 *
 */

#include <stdio.h>
#include <fcntl.h>      /* open */
#include "config.h"


int openp(const char *fpath, char *file, char *complete, int flags, int mode);

static int flgs, mod, value;
static const char *ftyp;
static FILE *fvalue;

static int
func(char *fnam)
{
	value = open(fnam, flgs, mod);
	return value < 0;
}

static int
ffunc(char *fnam)
{
	fvalue = fopen(fnam, ftyp);
	return fvalue == 0;
}

int
openp(const char *fpath, char *file, char *complete, int flags, int mode)
{
	flgs = flags;
	mod = mode;
	if (searchp(fpath, file, complete, func) < 0) {
		return -1;
	}
	return value;
}

FILE *
fopenp(const char *fpath, char *file, char *complete, const char *ftype)
{
	ftyp = ftype;
	if (searchp(fpath, file, complete, ffunc) < 0) {
		return 0;
	}
	return fvalue;
}
