/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 21, 2025.
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
#include "xdr_cssm.h"
#include "xdr_dldb.h"
#include <security_utilities/mach++.h>

bool_t
xdr_DLDbFlatIdentifier(XDR *xdrs, DataWalkers::DLDbFlatIdentifier *objp)
{
    if (!sec_xdr_pointer(xdrs, reinterpret_cast<uint8_t **>(&objp->uid), sizeof(CSSM_SUBSERVICE_UID), (xdrproc_t)xdr_CSSM_SUBSERVICE_UID))
        return (FALSE);
    if (!sec_xdr_charp(xdrs, reinterpret_cast<char **>(&objp->name), ~0))
        return (FALSE);
    if (!sec_xdr_pointer(xdrs, (uint8_t **)&objp->address, sizeof(CSSM_NET_ADDRESS), (xdrproc_t)xdr_CSSM_NET_ADDRESS))
        return (FALSE);
    return (TRUE);
}

bool_t
xdr_DLDbFlatIdentifierRef(XDR *xdrs, DataWalkers::DLDbFlatIdentifier **objp)
{
    if (!sec_xdr_reference(xdrs, reinterpret_cast<uint8_t **>(objp), sizeof(DataWalkers::DLDbFlatIdentifier), (xdrproc_t)xdr_DLDbFlatIdentifier))
        return (FALSE);
    return (TRUE);
}

CopyOut::~CopyOut()
{
	if (mData) {
		free(mData);
	}
	if(mDealloc && mSource) {
		MachPlusPlus::deallocate(mSource, mSourceLen);
	}
}
