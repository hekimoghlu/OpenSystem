/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 2, 2022.
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
#include "spray.h"
#ifndef lint
/*static char sccsid[] = "from: @(#)spray.x 1.2 87/09/18 Copyr 1987 Sun Micro";*/
/*static char sccsid[] = "from: @(#)spray.x	2.1 88/08/01 4.0 RPCSRC";*/
#endif /* not lint */
#include <sys/cdefs.h>
__RCSID("$FreeBSD: src/include/rpcsvc/spray.x,v 1.7 2003/05/04 02:51:42 obrien Exp $");

bool_t
xdr_spraytimeval(xdrs, objp)
	XDR *xdrs;
	spraytimeval *objp;
{

	if (!xdr_u_int(xdrs, &objp->sec))
		return (FALSE);
	if (!xdr_u_int(xdrs, &objp->usec))
		return (FALSE);
	return (TRUE);
}

bool_t
xdr_spraycumul(xdrs, objp)
	XDR *xdrs;
	spraycumul *objp;
{

	if (!xdr_u_int(xdrs, &objp->counter))
		return (FALSE);
	if (!xdr_spraytimeval(xdrs, &objp->clock))
		return (FALSE);
	return (TRUE);
}

bool_t
xdr_sprayarr(xdrs, objp)
	XDR *xdrs;
	sprayarr *objp;
{

	if (!xdr_bytes(xdrs, (char **)&objp->sprayarr_val, (u_int *)&objp->sprayarr_len, SPRAYMAX))
		return (FALSE);
	return (TRUE);
}
