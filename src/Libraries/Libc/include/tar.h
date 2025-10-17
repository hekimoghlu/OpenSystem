/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 9, 2023.
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
#ifndef _TAR_H
#define _TAR_H

#define	TMAGIC		"ustar"	/* ustar and a null */
#define	TMAGLEN		6
#define	TVERSION	"00"	/* 00 and no null */
#define	TVERSLEN	2

/* Values used in typeflag field */
#define	REGTYPE		'0'	/* Regular file */
#define	AREGTYPE	'\0'	/* Regular file */
#define	LNKTYPE		'1'	/* Link */
#define	SYMTYPE		'2'	/* Reserved */
#define	CHRTYPE		'3'	/* Character special */
#define	BLKTYPE		'4'	/* Block special */
#define	DIRTYPE		'5'	/* Directory */
#define	FIFOTYPE	'6'	/* FIFO special */
#define	CONTTYPE	'7'	/* Reserved */

/* Bits used in the mode field - values in octal */
#define	TSUID		04000	/* Set UID on execution */
#define	TSGID		02000	/* Set GID on execution */
#define	TSVTX		01000	/* Reserved */
				/* File permissions */
#define	TUREAD		00400	/* Read by owner */
#define	TUWRITE		00200	/* Write by owner */
#define	TUEXEC		00100	/* Execute/Search by owner */
#define	TGREAD		00040	/* Read by group */
#define	TGWRITE		00020	/* Write by group */
#define	TGEXEC		00010	/* Execute/Search by group */
#define	TOREAD		00004	/* Read by other */
#define	TOWRITE		00002	/* Write by other */
#define	TOEXEC		00001	/* Execute/Search by other */

#endif
