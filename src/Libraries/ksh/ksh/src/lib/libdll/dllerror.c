/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, November 17, 2023.
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
 */

#include "dlllib.h"

Dllstate_t	state;

/*
 * return error message from last failed dl*() call
 * retain==0 resets the last dl*() error
 */

extern char*
dllerror(int retain)
{
	char*	s;

	if (state.error)
	{
		state.error = retain;
		return state.errorbuf;
	}
	s = dlerror();
	if (retain)
	{
		state.error = retain;
		sfsprintf(state.errorbuf, sizeof(state.errorbuf), "%s", s);
	}
	return s;
}
