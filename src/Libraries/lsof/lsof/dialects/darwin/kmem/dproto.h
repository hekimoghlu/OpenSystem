/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 17, 2025.
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
 * Copyright 1994 Purdue Research Foundation, West Lafayette, Indiana
 * 47907.  All rights reserved.
 *
 * Written by Victor A. Abell
 *
 * This software is not subject to any license of the American Telephone
 * and Telegraph Company or the Regents of the University of California.
 *
 * Permission is granted to anyone to use this software for any purpose on
 * any computer system, and to alter it and redistribute it freely, subject
 * to the following restrictions:
 *
 * 1. Neither the authors nor Purdue University are responsible for any
 *    consequences of the use of this software.
 *
 * 2. The origin of this software must not be misrepresented, either by
 *    explicit claim or by omission.  Credit to the authors and Purdue
 *    University must appear in documentation and sources.
 *
 * 3. Altered versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 *
 * 4. This notice may not be removed or altered.
 */


/*
 * $Id: dproto.h,v 1.4 2005/11/01 20:24:51 abe Exp $
 */

_PROTOTYPE(extern int is_file_named,(char *p, int cd));
_PROTOTYPE(extern struct l_vfs *readvfs,(KA_T vm));

#if	defined(HASKQUEUE)
_PROTOTYPE(extern void process_kqueue,(KA_T ka));
#endif	/* defined(HASKQUEUE) */

#if	defined(HASPIPEFN)
_PROTOTYPE(extern void process_pipe,(KA_T pa));
#endif	/* defined(HASPIPEFN) */

#if	defined(HASPSXSEM)
_PROTOTYPE(extern void process_psxsem,(KA_T pa));
#endif	/* defined(HASPSXSEM) */

#if	defined(HASPSXSHM)
_PROTOTYPE(extern void process_psxshm,(KA_T pa));
#endif	/* defined(HASPSXSHM) */

#if	defined(HAS9660FS)
_PROTOTYPE(extern int read_iso_node,(struct vnode *v, dev_t *d, int *dd, INODETYPE *ino, long *nl, SZOFFTYPE *sz));
#endif	/* defined(HAS9660FS) */

_PROTOTYPE(extern void process_socket,(KA_T sa));
