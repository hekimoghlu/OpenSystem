/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 9, 2023.
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
#pragma prototyped

/*
 * AT&T Research
 *
 * common dirent maintenance interface
 */

#ifndef _AST_DIR_H
#define _AST_DIR_H

#include <ast_lib.h>

#if _mem_d_fileno_dirent || _mem_d_ino_dirent
#if !_mem_d_fileno_dirent
#undef	_mem_d_fileno_dirent
#define _mem_d_fileno_dirent	1
#define d_fileno		d_ino
#endif
#endif

#if _BLD_ast
#include "dirlib.h"
#else
#include <dirent.h>
#endif

#if _mem_d_fileno_dirent
#define D_FILENO(d)		((d)->d_fileno)
#endif

#if _mem_d_namlen_dirent
#define D_NAMLEN(d)		((d)->d_namlen)
#else
#define D_NAMLEN(d)		(strlen((d)->d_name))
#endif

#if _mem_d_reclen_dirent
#define D_RECLEN(d)		((d)->d_reclen)
#else
#define D_RECLEN(d)		D_RECSIZ(d,D_NAMLEN(d))
#endif

#define D_RECSIZ(d,n)		(sizeof(*(d))-sizeof((d)->d_name)+((n)+sizeof(char*))&~(sizeof(char*)-1))

/*
 * NOTE: 2003-03-27 mac osx bug symlink==DT_REG bug discovered;
 *	 the kernel *and* all directories must be fixed, so d_type
 *	 is summarily disabled until we see that happen
 */

#if _mem_d_type_dirent && defined(DT_UNKNOWN) && defined(DT_REG) && defined(DT_DIR) && defined(DT_LNK) && ! ( __APPLE__ || __MACH__ )
#define D_TYPE(d)		((d)->d_type)
#endif

#endif
