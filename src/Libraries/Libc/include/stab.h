/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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
 * The following are symbols used by various debuggers and by the Pascal
 * compiler.  Each of them must have one (or more) of the bits defined by
 * the N_STAB mask set.
 */

#define	N_GSYM		0x20	/* global symbol */
#define	N_FNAME		0x22	/* F77 function name */
#define	N_FUN		0x24	/* procedure name */
#define	N_STSYM		0x26	/* data segment variable */
#define	N_LCSYM		0x28	/* bss segment variable */
#define	N_MAIN		0x2a	/* main function name */
#define	N_PC		0x30	/* global Pascal symbol */
#define	N_RSYM		0x40	/* register variable */
#define	N_SLINE		0x44	/* text segment line number */
#define	N_DSLINE	0x46	/* data segment line number */
#define	N_BSLINE	0x48	/* bss segment line number */
#define	N_SSYM		0x60	/* structure/union element */
#define	N_SO		0x64	/* main source file name */
#define	N_LSYM		0x80	/* stack variable */
#define	N_BINCL		0x82	/* include file beginning */
#define	N_SOL		0x84	/* included source file name */
#define	N_PSYM		0xa0	/* parameter variable */
#define	N_EINCL		0xa2	/* include file end */
#define	N_ENTRY		0xa4	/* alternate entry point */
#define	N_LBRAC		0xc0	/* left bracket */
#define	N_EXCL		0xc2	/* deleted include file */
#define	N_RBRAC		0xe0	/* right bracket */
#define	N_BCOMM		0xe2	/* begin common */
#define	N_ECOMM		0xe4	/* end common */
#define	N_ECOML		0xe8	/* end common (local name) */
#define	N_LENG		0xfe	/* length of preceding entry */
