/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 26, 2023.
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

/*	Close a dictionary
**
**	Written by Kiem-Phong Vo (11/15/2010)
*/
#if __STD_C
int dtclose(Dt_t* dt)
#else
int dtclose(dt)
Dt_t*	dt;
#endif
{
	int		ev, type;
	Dt_t		pdt;
	Dtdisc_t	*disc = dt->disc;

	if(!dt || dt->nview > 0 ) /* can't close if being viewed */
		return -1;

	if(disc && disc->eventf) /* announce closing event */
		ev = (*disc->eventf)(dt, DT_CLOSE, (Void_t*)1, disc);
	else	ev = 0;
	if(ev < 0) /* cannot close */
		return -1;

	if(dt->view) /* turn off viewing at this point */
		dtview(dt,NIL(Dt_t*));

	type = dt->data->type; /* save before memory is freed */
	memcpy(&pdt, dt, sizeof(Dt_t));

	if(ev == 0 ) /* release all allocated data */
	{	(void)(*(dt->meth->searchf))(dt,NIL(Void_t*),DT_CLEAR);
		(void)(*dt->meth->eventf)(dt, DT_CLOSE, (Void_t*)0);
		/**/DEBUG_ASSERT(!dt->data);
	}
	if(!(type&DT_INDATA) )
		(void)free(dt);

	if(disc && disc->eventf) /* announce end of closing activities */
		(void)(*disc->eventf)(&pdt, DT_ENDCLOSE, (Void_t*)0, disc);

	return 0;
}
