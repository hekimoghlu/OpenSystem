/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 25, 2024.
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
 * Ksh - AT&T Labs
 * Written by David Korn
 * This file defines all the  read/write shell global variables
 */

#include	"defs.h"
#include	"jobs.h"
#include	"shlex.h"
#include	"edit.h"
#include	"timeout.h"

Shell_t			sh = {0};
struct	shared		*shgd;
#ifdef	__IMPORT__
    Shell_t		*_imp__sh = &sh;
#endif

Dtdisc_t	_Nvdisc =
{
	offsetof(Namval_t,nvname), -1 , 0, 0, 0, nv_compare
};

/* reserve room for writable state table */
char *sh_lexstates[ST_NONE] = {0};

struct jobs	job = {0};
int32_t		sh_mailchk = 600;

