/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 6, 2025.
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
 * return p stat info
 */

#include "reglib.h"

regstat_t*
regstat(const regex_t* p)
{
	register Rex_t*	e;

	p->env->stats.re_flags = p->env->flags;
	p->env->stats.re_info = 0;
	e = p->env->rex;
	if (e && e->type == REX_BM)
	{
		p->env->stats.re_record = p->env->rex->re.bm.size;
		e = e->next;
	}
	else
		p->env->stats.re_record = 0;
	if (e && e->type == REX_BEG)
		e = e->next;
	if (e && e->type == REX_STRING)
		e = e->next;
	if (!e || e->type == REX_END && !e->next)
		p->env->stats.re_info |= REG_LITERAL;
	p->env->stats.re_record = (p && p->env && p->env->rex->type == REX_BM) ? p->env->rex->re.bm.size : -1;
	return &p->env->stats;
}
