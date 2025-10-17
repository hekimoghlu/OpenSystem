/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 17, 2022.
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
 * Portions Copyright 2005 Apple Computer, Inc.  All rights reserved.
 *
 * Copyright 2005 Purdue Research Foundation, West Lafayette, Indiana
 * 47907.  All rights reserved.
 *
 * Written by Allan Nathanson, Apple Computer, Inc., and Victor A.
 * Abell, Purdue University.
 *
 * This software is not subject to any license of the American Telephone
 * and Telegraph Company or the Regents of the University of California.
 *
 * Permission is granted to anyone to use this software for any purpose on
 * any computer system, and to alter it and redistribute it freely, subject
 * to the following restrictions:
 *
 * 1. Neither the authors, nor Apple Computer, Inc. nor Purdue University
 *    are responsible for any consequences of the use of this software.
 *
 * 2. The origin of this software must not be misrepresented, either
 *    by explicit claim or by omission.  Credit to the authors, Apple
 *    Computer, Inc. and Purdue University must appear in documentation
 *    and sources.
 *
 * 3. Altered versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 *
 * 4. This notice may not be removed or altered.
 */


/*
 * $Id: dproto.h,v 1.6 2012/04/10 16:41:04 abe Exp abe $
 */

_PROTOTYPE(extern void enter_file_info,(struct proc_fileinfo *pfi));
_PROTOTYPE(extern void enter_vnode_info,(struct vnode_info_path *vip));
_PROTOTYPE(extern void err2nm,(char *pfx));
_PROTOTYPE(extern int is_file_named,(char *p, int cd));
_PROTOTYPE(extern void process_atalk,(int pid, int32_t fd));
_PROTOTYPE(extern void process_fsevents,(int pid, int32_t fd));
_PROTOTYPE(extern void process_kqueue,(int pid, int32_t fd));
_PROTOTYPE(extern void process_pipe,(int pid, int32_t fd));
_PROTOTYPE(extern void process_psem,(int pid, int32_t fd));
_PROTOTYPE(extern void process_pshm,(int pid, int32_t fd));
_PROTOTYPE(extern void process_socket,(int pid, int32_t fd));
_PROTOTYPE(extern void process_vnode,(int pid, int32_t fd));
_PROTOTYPE(extern void process_netpolicy,(int pid, int32_t fd));
#ifdef	PROC_PIDLISTFILEPORTS
_PROTOTYPE(extern void process_fileport_pipe,(int pid, uint32_t fileport));
_PROTOTYPE(extern void process_fileport_pshm,(int pid, uint32_t fileport));
_PROTOTYPE(extern void process_fileport_socket,(int pid, uint32_t fileport));
_PROTOTYPE(extern void process_fileport_vnode,(int pid, uint32_t fileport));
#endif	/* PROC_PIDLISTFILEPORTS */
#ifdef	PROC_FP_GUARDED
extern struct pff_tab Pgf_tab[];
#endif	/* PROC_FP_GUARDED */

