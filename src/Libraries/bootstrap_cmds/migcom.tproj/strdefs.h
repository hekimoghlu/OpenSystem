/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 11, 2021.
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
 * Mach Operating System
 * Copyright (c) 1991,1990 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS
 * CONDITION.  CARNEGIE MELLON DISCLAIMS ANY LIABILITY OF ANY KIND FOR
 * ANY DAMAGES WHATSOEVER RESULTING FROM THE USE OF THIS SOFTWARE.
 *
 * Carnegie Mellon requests users of this software to return to
 *
 *  Software Distribution Coordinator  or  Software.Distribution@CS.CMU.EDU
 *  School of Computer Science
 *  Carnegie Mellon University
 *  Pittsburgh PA 15213-3890
 *
 * any improvements or extensions that they make and grant Carnegie the
 * rights to redistribute these changes.
 */
/*
 * 91/02/05  17:55:57  mrt
 *  Changed to new Mach copyright
 *  [91/02/01  17:56:03  mrt]
 *
 * 90/06/02  15:05:49  rpd
 *  Created for new IPC.
 *  [90/03/26  21:13:56  rpd]
 *
 * 07-Apr-89  Richard Draves (rpd) at Carnegie-Mellon University
 *  Extensive revamping.  Added polymorphic arguments.
 *  Allow multiple variable-sized inline arguments in messages.
 *
 * 15-Jun-87  David Black (dlb) at Carnegie-Mellon University
 *  Fixed strNULL to be the null string instead of the null string
 *  pointer.
 *
 * 27-May-87  Richard Draves (rpd) at Carnegie-Mellon University
 *  Created.
 */

#ifndef STRDEFS_H
#define STRDEFS_H

#include <string.h>
#include <mach/boolean.h>

typedef char *string_t;
typedef string_t identifier_t;

#define MAX_STR_LEN 200

#define strNULL ((string_t) 0)

extern string_t strmake( char *string );
extern string_t strconcat( string_t left, string_t right );
extern string_t strphrase( string_t left, string_t right );
extern void strfree( string_t string );

#define streql(a, b)  (strcmp((a), (b)) == 0)

extern char *strbool( boolean_t bool );
extern char *strstring( string_t string );
extern char *toupperstr( char *string );

#endif  /* STRDEFS_H */
