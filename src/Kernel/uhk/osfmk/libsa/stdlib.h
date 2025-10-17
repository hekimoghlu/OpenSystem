/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 11, 2025.
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
 * @OSF_COPYRIGHT@
 *
 */
/*
 * HISTORY
 *
 * Revision 1.1.1.1  1998/09/22 21:05:51  wsanchez
 * Import of Mac OS X kernel (~semeria)
 *
 * Revision 1.1.1.1  1998/03/07 02:25:35  wsanchez
 * Import of OSF Mach kernel (~mburg)
 *
 * Revision 1.1.4.1  1997/02/21  15:43:19  barbou
 *      Removed "size_t" definition, include "types.h" instead.
 *      [1997/02/21  15:36:24  barbou]
 *
 * Revision 1.1.2.3  1996/09/30  10:14:34  bruel
 *      Added strtol and strtoul prototypes.
 *      [96/09/30            bruel]
 *
 * Revision 1.1.2.2  1996/09/23  15:06:22  bruel
 *      removed bzero and bcopy definitions.
 *      [96/09/23            bruel]
 *
 * Revision 1.1.2.1  1996/09/17  16:56:24  bruel
 *      created from standalone mach servers.
 *      [96/09/17            bruel]
 *
 * $EndLog$
 */

#ifndef _MACH_STDLIB_H_
#define _MACH_STDLIB_H_

#include <types.h>

#ifndef NULL
#define NULL    (void *)0
#endif

extern int     atoi(const char *);
extern char    *itoa(int, char *);

extern void     free(void *);
extern void     *malloc(size_t);
extern void     *realloc(void *, size_t);

extern char     *getenv(const char *);

extern void     exit(int);

extern long int strtol(const char *, char **, int);
extern unsigned long int strtoul(const char *, char **, int);

#endif /* _MACH_STDLIB_H_ */
