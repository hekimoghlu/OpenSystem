/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 15, 2022.
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
 * Glenn Fowler
 * AT&T Research
 *
 * mode_t common definitions
 */

#ifndef _MODELIB_H
#define _MODELIB_H

#include <ast.h>
#include <ls.h>
#include <modex.h>

#define MODELEN	10
#define PERMLEN	24

#define modetab	_mode_table_	/* data hiding				*/
#define permmap	_mode_permmap_	/* data hiding				*/

struct modeop			/* ops for each char in mode string	*/
{
	int	mask1;		/* first mask				*/
	int	shift1;		/* first shift count			*/
	int	mask2;		/* second mask				*/
	int	shift2;		/* second shift count			*/
	char*	name;		/* mode char using mask/shift as index	*/
};

extern struct modeop	modetab[];
extern int		permmap[];

#endif
