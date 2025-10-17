/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 8, 2025.
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
#include "xdr_auth.h"

#include <unistd.h>
#include <stdlib.h>
#include <string.h>

bool_t
xdr_AuthorizationItem(XDR *xdrs, AuthorizationItem *objp)
{
    if (!sec_xdr_charp(xdrs, (char **)&objp->name, ~0))
		return (FALSE);
		
    u_int valueLength;
	
    if (xdrs->x_op == XDR_ENCODE) {
		if (objp->valueLength > (u_int)~0)
			return (FALSE);
		valueLength = (u_int)objp->valueLength;
    }
	
    if (!sec_xdr_bytes(xdrs, (uint8_t **)&objp->value, &valueLength, ~0))
		return (FALSE);
		
    if (xdrs->x_op == XDR_DECODE)
		objp->valueLength = valueLength;
	
	// This is only ever 32 bits, but prototyped with long on 32 bit and int on 64 bit to fall in line with UInt32
    if (!xdr_u_long(xdrs, &objp->flags))
		return (FALSE);
		
    return (TRUE);
}

bool_t
xdr_AuthorizationItemSet(XDR *xdrs, AuthorizationItemSet *objp)
{
    return sec_xdr_array(xdrs, (uint8_t **)&objp->items, (u_int *)&objp->count, ~0, sizeof(AuthorizationItem), (xdrproc_t)xdr_AuthorizationItem);
}

bool_t
xdr_AuthorizationItemSetPtr(XDR *xdrs, AuthorizationItemSet **objp)
{
	return sec_xdr_reference(xdrs, (uint8_t **)objp,sizeof(AuthorizationItemSet), (xdrproc_t)xdr_AuthorizationItemSet);
}

inline bool_t copyin_AuthorizationItemSet(const AuthorizationItemSet *rights, void **copy, mach_msg_size_t *size)
{
	return copyin((AuthorizationItemSet *)rights, (xdrproc_t)xdr_AuthorizationItemSet, copy, size);
}

inline bool_t copyout_AuthorizationItemSet(const void *copy, mach_msg_size_t size, AuthorizationItemSet **rights)
{
	u_int length = 0;
	void *data = NULL; // allocate data for us
	bool_t ret = copyout(copy, size, (xdrproc_t)xdr_AuthorizationItemSetPtr, &data, &length);
	if (ret)
		*rights = data;
    return ret;
}
