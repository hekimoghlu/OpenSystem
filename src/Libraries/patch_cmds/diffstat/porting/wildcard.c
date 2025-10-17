/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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

#ifndef NO_IDENT
static char *Id = "$Id: wildcard.c,v 1.1 2004/05/03 13:12:10 tom Exp $";
#endif

/*
 * wildcard.c - perform wildcard expansion for non-UNIX configurations
 */

#include "system.h"

#if HAVE_STDLIB_H
# include <stdlib.h>
#endif

#if HAVE_STRING_H
# include <string.h>
#endif

#if SYS_MSDOS || SYS_OS2
# if SYS_MSDOS
#  include <dos.h>
#  include <dir.h>		/* defines MAXPATH */
#  define FILENAME_MAX MAXPATH
# endif
# define DeclareFind(p)		struct find_t p
# define DirEntryStr(p)		p.name
# define DirFindFirst(path,p)	(!_dos_findfirst(path, 0, &p))
# define DirFindNext(p)		(!_dos_findnext(&p))
#endif

#if SYS_VMS

#include <starlet.h>		/* DEC-C (e.g., sys$parse) */
#include <stdio.h>		/* perror */

#include <rms.h>
#include <descrip.h>

#endif /* SYS_VMS */

int
has_wildcard(char *path)
{
#if SYS_VMS
    return (strstr(path, "...") != 0
	    || strchr(path, '*') != 0
	    || strchr(path, '?') != 0);
#else /* SYS_MSDOS, SYS_OS2 */
    return (strchr(path, '*') != 0
	    || strchr(path, '?') != 0);
#endif
}

int
expand_wildcard(char *path, int initiate)
{
#if SYS_MSDOS || SYS_OS2
    static DeclareFind(p);
    static char temp[FILENAME_MAX + 1];
    register char *leaf;

    if ((leaf = strchr(path, '/')) == 0
	&& (leaf = strchr(path, '\\')) == 0)
	leaf = path;
    else
	leaf++;

    if ((initiate && DirFindFirst(strcpy(temp, path), p))
	|| DirFindNext(p)) {
	(void) strcpy(leaf, DirEntryStr(p));
	return TRUE;
    }
#endif /* SYS_MSDOS || SYS_OS2 */
#if SYS_VMS
    static struct FAB zfab;
    static struct NAM znam;
    static char my_esa[NAM$C_MAXRSS];	/* expanded: SYS$PARSE */
    static char my_rsa[NAM$C_MAXRSS];	/* expanded: SYS$SEARCH */

    if (initiate) {
	zfab = cc$rms_fab;
	zfab.fab$l_fop = FAB$M_NAM;
	zfab.fab$l_nam = &znam;	/* FAB => NAM block     */
	zfab.fab$l_dna = "*.*;";	/* Default-selection    */
	zfab.fab$b_dns = strlen(zfab.fab$l_dna);

	zfab.fab$l_fna = path;
	zfab.fab$b_fns = strlen(path);

	znam = cc$rms_nam;
	znam.nam$b_ess = sizeof(my_esa);
	znam.nam$l_esa = my_esa;
	znam.nam$b_rss = sizeof(my_rsa);
	znam.nam$l_rsa = my_rsa;

	if (sys$parse(&zfab) != RMS$_NORMAL) {
	    perror(path);
	    exit(EXIT_FAILURE);
	}
    }
    if (sys$search(&zfab) == RMS$_NORMAL) {
	strncpy(path, my_rsa, znam.nam$b_rsl)[znam.nam$b_rsl] = '\0';
	return (TRUE);
    }
#endif /* SYS_VMS */
#if SYS_MSDOS
#endif
    return FALSE;
}
