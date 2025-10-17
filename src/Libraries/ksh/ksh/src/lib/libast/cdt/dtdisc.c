/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 14, 2023.
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
#include	"dthdr.h"

/*	Change discipline.
**	dt :	dictionary
**	disc :	discipline
**
**	Written by Kiem-Phong Vo (5/26/96)
*/

#if __STD_C
static Void_t* dtmemory(Dt_t* dt, Void_t* addr, size_t size, Dtdisc_t* disc)
#else
static Void_t* dtmemory(dt, addr, size, disc)
Dt_t* 		dt;	/* dictionary			*/
Void_t* 	addr;	/* address to be manipulate	*/
size_t		size;	/* size to obtain		*/
Dtdisc_t* 	disc;	/* discipline			*/
#endif
{
	if(addr)
	{	if(size == 0)
		{	free(addr);
			return NIL(Void_t*);
		}
		else	return realloc(addr,size);
	}
	else	return size > 0 ? malloc(size) : NIL(Void_t*);
}

#if __STD_C
Dtdisc_t* dtdisc(Dt_t* dt, Dtdisc_t* disc, int type)
#else
Dtdisc_t* dtdisc(dt,disc,type)
Dt_t*		dt;
Dtdisc_t*	disc;
int		type;
#endif
{
	Dtdisc_t	*old;
	Dtlink_t	*list;

	if(!(old = dt->disc) )	/* initialization call from dtopen() */
	{	dt->disc = disc;
		if(!(dt->memoryf = disc->memoryf) )
			dt->memoryf = dtmemory;
		return disc;
	}

	if(!disc) /* only want to know current discipline */
		return old;

	if(old->eventf && (*old->eventf)(dt,DT_DISC,(Void_t*)disc,old) < 0)
		return NIL(Dtdisc_t*);

	if((type & (DT_SAMEHASH|DT_SAMECMP)) != (DT_SAMEHASH|DT_SAMECMP) )
		list = dtextract(dt); /* grab the list of objects if any */
	else	list = NIL(Dtlink_t*);

	dt->disc = disc;
	if(!(dt->memoryf = disc->memoryf) )
		dt->memoryf = dtmemory;

	if(list ) /* reinsert extracted objects (with new discipline) */
		dtrestore(dt, list);

	return old;
}
