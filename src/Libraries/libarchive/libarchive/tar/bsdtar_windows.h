/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 16, 2024.
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
#ifndef BSDTAR_WINDOWS_H
#define	BSDTAR_WINDOWS_H 1
#include <direct.h>
#include <windows.h>
#include <io.h>
#include <fcntl.h>

#ifndef PRId64
#define	PRId64 "I64"
#endif
#define	geteuid()	0

#ifndef __WATCOMC__

#ifndef S_IFIFO
#define	S_IFIFO	0010000 /* pipe */
#endif

#include <string.h>  /* Must include before redefining 'strdup' */
#if !defined(__BORLANDC__)
#define	strdup _strdup
#endif
#if !defined(__BORLANDC__)
#define	getcwd _getcwd
#endif

#define	chdir __tar_chdir
int __tar_chdir(const char *);

#ifndef S_ISREG
#define	S_ISREG(a)	(a & _S_IFREG)
#endif
#ifndef S_ISBLK
#define	S_ISBLK(a)	(0)
#endif

#endif

#endif /* BSDTAR_WINDOWS_H */
