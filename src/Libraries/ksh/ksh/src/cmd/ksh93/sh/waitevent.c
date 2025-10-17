/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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

#include	"defs.h"
/*
 *  This installs a hook to allow the processing of events when
 *  the shell is waiting for input and when the shell is
 *  waiting for job completion.
 *  The previous waitevent hook function is returned
 */


void	*sh_waitnotify(int(*newevent)(int,long,int))
{
	int (*old)(int,long,int);
	old = shgd->waitevent;
	shgd->waitevent = newevent;
	return((void*)old);
}

#if __OBSOLETE__ < 20080101
/*
 * this used to be a private symbol
 * retain the old name for a bit for a smooth transition
 */

#if defined(__EXPORT__)
#define extern		__EXPORT__
#endif

extern void	*_sh_waitnotify(int(*newevent)(int,long,int))
{
	return sh_waitnotify(newevent);
}

#endif
