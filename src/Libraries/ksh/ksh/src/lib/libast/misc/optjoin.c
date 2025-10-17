/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 11, 2024.
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
 * multi-pass commmand line option parse assist
 *
 *	int fun(char** argv, int last)
 *
 * each fun() argument parses as much of argv as
 * possible starting at (opt_info.index,opt_info.offset) using
 * optget()
 *
 * if last!=0 then fun is the last pass to view
 * the current arg, otherwise fun sets opt_info.again=1
 * and another pass will get a crack at it
 *
 * 0 fun() return causes immediate optjoin() 0 return
 *
 * optjoin() returns non-zero if more args remain
 * to be parsed at opt_info.index
 */

#include <optlib.h>

typedef int (*Optpass_f)(char**, int);

int
optjoin(char** argv, ...)
{
	va_list			ap;
	register Optpass_f	fun;
	register Optpass_f	rep;
	Optpass_f		err;
	Optstate_t*		state;
	int			r;
	int			more;
	int			user;
	int			last_index;
	int			last_offset;
	int			err_index;
	int			err_offset;

	state = optstate(&opt_info);
	err = rep = 0;
	r = -1;
	while (r < 0)
	{
		va_start(ap, argv);
		state->join = 0;
		while (fun = va_arg(ap, Optpass_f))
		{
			last_index = opt_info.index;
			last_offset = opt_info.offset;
			state->join++;
			user = (*fun)(argv, 0);
			more = argv[opt_info.index] != 0;
			if (!opt_info.again)
			{
				if (!more)
				{
					state->join = 0;
					r = 0;
					break;
				}
				if (!user)
				{
					if (*argv[opt_info.index] != '+')
					{
						state->join = 0;
						r = 1;
						break;
					}
					opt_info.again = -1;
				}
				else
					err = 0;
			}
			if (opt_info.again)
			{
				if (opt_info.again > 0 && (!err || err_index < opt_info.index || err_index == opt_info.index && err_offset < opt_info.offset))
				{
					err = fun;
					err_index = opt_info.index;
					err_offset = opt_info.offset;
				}
				opt_info.again = 0;
				opt_info.index = state->pindex ? state->pindex : 1;
				opt_info.offset = state->poffset;
			}
			if (!rep || opt_info.index != last_index || opt_info.offset != last_offset)
				rep = fun;
			else if (fun == rep)
			{
				if (!err)
				{
					state->join = 0;
					r = 1;
					break;
				}
				(*err)(argv, 1);
				opt_info.offset = 0;
			}
		}
		va_end(ap);
	}
	return r;
}
