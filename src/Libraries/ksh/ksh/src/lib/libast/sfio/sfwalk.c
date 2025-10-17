/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
#include	"sfhdr.h"

/* Walk streams and run operations on them
**
** Written by Kiem-Phong Vo.
*/

#if __STD_C
int sfwalk(Sfwalk_f walkf, Void_t* data, int type)
#else
int sfwalk(walkf, data, type)
Sfwalk_f	walkf;	/* return <0: stop, >=0: continue	*/
Void_t*		data;
int		type;	/* walk streams with all given flags	*/
#endif
{
	Sfpool_t	*p;
	Sfio_t		*f;
	int		n, rv;

	/* truly initializing std-streams before walking */
	if(sfstdin->mode & SF_INIT)
		_sfmode(sfstdin, (sfstdin->mode & SF_RDWR), 0);
	if(sfstdout->mode & SF_INIT)
		_sfmode(sfstdout, (sfstdout->mode & SF_RDWR), 0);
	if(sfstderr->mode & SF_INIT)
		_sfmode(sfstderr, (sfstderr->mode & SF_RDWR), 0);

	for(rv = 0, p = &_Sfpool; p; p = p->next)
	{	for(n = 0; n < p->n_sf; )
		{	f = p->sf[n];

			if(type != 0 && (f->_flags&type) != type )
				continue; /* not in the interested set */

			if((rv = (*walkf)(f, data)) < 0)
				return rv;

			if(p->sf[n] == f) /* move forward to next stream */
				n += 1;
			/* else - a sfclose() was done on current stream */
		}
	}

	return rv;
}
