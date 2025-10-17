/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 7, 2025.
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
/* XXX THIS HAS BEEN MODIFIED FOR INCORPORATION INTO BASH XXX */

#ifndef _SH_GETOPT_H
#define _SH_GETOPT_H 1

#include "stdc.h"

/* For communication from `getopt' to the caller.
   When `getopt' finds an option that takes an argument,
   the argument value is returned here.
   Also, when `ordering' is RETURN_IN_ORDER,
   each non-option ARGV-element is returned here.  */

extern char *sh_optarg;

/* Index in ARGV of the next element to be scanned.
   This is used for communication to and from the caller
   and for communication between successive calls to `getopt'.

   On entry to `getopt', zero means this is the first call; initialize.

   When `getopt' returns EOF, this is the index of the first of the
   non-option elements that the caller should itself scan.

   Otherwise, `sh_optind' communicates from one call to the next
   how much of ARGV has been scanned so far.  */

extern int sh_optind;

/* Callers store zero here to inhibit the error message `getopt' prints
   for unrecognized options.  */

extern int sh_opterr;

/* Set to an option character which was unrecognized.  */

extern int sh_optopt;

/* Set to 1 when an unrecognized option is encountered. */
extern int sh_badopt;

extern int sh_getopt __P((int, char *const *, const char *));
extern void sh_getopt_restore_state __P((char **));

#endif /* _SH_GETOPT_H */
