/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 26, 2022.
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

/*	Change search method.
**
**	Written by Kiem-Phong Vo (05/25/96)
*/

#if __STD_C
Dtmethod_t* dtmethod(Dt_t* dt, Dtmethod_t* meth)
#else
Dtmethod_t* dtmethod(dt, meth)
Dt_t*		dt;
Dtmethod_t*	meth;
#endif
{
	Dtlink_t	*list;
	Dtdisc_t	*disc = dt->disc;
	Dtmethod_t	*oldmt = dt->meth;
	Dtdata_t	*newdt, *olddt = dt->data;

	if(!meth || meth == oldmt)
		return oldmt;

	/* ask discipline if switching to new method is ok */
	if(disc->eventf && (*disc->eventf)(dt,DT_METH,(Void_t*)meth,disc) < 0)
		return NIL(Dtmethod_t*);

	list = dtextract(dt); /* extract elements out of dictionary */

	/* try to create internal structure for new method */
	if(dt->searchf == oldmt->searchf) /* ie, not viewpathing */
		dt->searchf = meth->searchf;
	dt->meth = meth;
	dt->data = NIL(Dtdata_t*);
	if((*dt->meth->eventf)(dt, DT_OPEN, NIL(Void_t*)) < 0 )
		newdt = NIL(Dtdata_t*);
	else	newdt = dt->data;

	/* see what need to be done to data of the old method */ 
	if(dt->searchf == meth->searchf)
		dt->searchf = oldmt->searchf;
	dt->meth = oldmt;
	dt->data = olddt;
	if(newdt) /* switch was successful, remove old data */
	{	(void)(*dt->meth->eventf)(dt, DT_CLOSE, NIL(Void_t*));

		if(dt->searchf == oldmt->searchf)
			dt->searchf = meth->searchf;
		dt->meth = meth;
		dt->data = newdt;
		dtrestore(dt, list);
		return oldmt;
	}
	else /* switch failed, restore dictionary to previous states */
	{	dtrestore(dt, list); 
		return NIL(Dtmethod_t*);
	}
}

/* customize certain actions in a container data structure */
int dtcustomize(Dt_t* dt, int type, int action)
{
	int	done = 0;

	if((type&DT_SHARE) &&
	   (!dt->meth->eventf || (*dt->meth->eventf)(dt, DT_SHARE, (Void_t*)((long)action)) >= 0) )
	{	if(action <= 0 )
			dt->data->type &= ~DT_SHARE;
		else	dt->data->type |=  DT_SHARE;
		done |= DT_SHARE;
	}

	if((type&DT_ANNOUNCE) &&
	   (!dt->meth->eventf || (*dt->meth->eventf)(dt, DT_ANNOUNCE, (Void_t*)((long)action)) >= 0) )
	{	if(action <= 0 )
			dt->data->type &= ~DT_ANNOUNCE;
		else	dt->data->type |=  DT_ANNOUNCE;
		done |= DT_ANNOUNCE;
	}

	if((type&DT_OPTIMIZE) &&
	   (!dt->meth->eventf || (*dt->meth->eventf)(dt, DT_OPTIMIZE, (Void_t*)((long)action)) >= 0) )
		done |= DT_OPTIMIZE;

	return done;
}
