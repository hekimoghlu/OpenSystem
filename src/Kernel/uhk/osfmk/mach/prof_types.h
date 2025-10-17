/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 12, 2024.
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
 * @OSF_FREE_COPYRIGHT@
 */
/*
 * HISTORY
 *
 * Revision 1.1.1.1  1998/09/22 21:05:30  wsanchez
 * Import of Mac OS X kernel (~semeria)
 *
 * Revision 1.1.1.1  1998/03/07 02:25:46  wsanchez
 * Import of OSF Mach kernel (~mburg)
 *
 * Revision 1.2.7.2  1995/01/26  22:15:46  ezf
 *      corrected CR
 *      [1995/01/26  21:16:02  ezf]
 *
 * Revision 1.2.3.2  1993/06/09  02:43:16  gm
 *      Added to OSF/1 R1.3 from NMK15.0.
 *      [1993/06/02  21:18:04  jeffc]
 *
 * Revision 1.2  1993/04/19  16:39:03  devrcs
 *      ansi C conformance changes
 *      [1993/02/02  18:54:26  david]
 *
 * Revision 1.1  1992/09/30  02:32:04  robert
 *      Initial revision
 *
 * $EndLog$
 */
/*
 * Mach Operating System
 * Copyright (c) 1991,1990,1989 Carnegie Mellon University
 * All Rights Reserved.
 *
 * Permission to use, copy, modify and distribute this software and its
 * documentation is hereby granted, provided that both the copyright
 * notice and this permission notice appear in all copies of the
 * software, derivative works or modified versions, and any portions
 * thereof, and that both notices appear in supporting documentation.
 *
 * CARNEGIE MELLON ALLOWS FREE USE OF THIS SOFTWARE IN ITS "AS IS"
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
 * any improvements or extensions that they make and grant Carnegie Mellon
 * the rights to redistribute these changes.
 */
/*
 */

#ifndef _MACH_PROF_TYPES_H
#define _MACH_PROF_TYPES_H

#define SAMPLE_MAX      256     /* Max array size */
typedef unsigned        sample_array_t[SAMPLE_MAX];

#endif  /* _MACH_PROF_TYPES_H */
