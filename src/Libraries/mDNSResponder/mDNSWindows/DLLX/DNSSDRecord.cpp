/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 19, 2023.
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
#include "stdafx.h"

#include "DNSSDRecord.h"

#include "StringServices.h"

#include <DebugServices.h>





// CDNSSDRecord



STDMETHODIMP CDNSSDRecord::Update(DNSSDFlags flags, VARIANT rdata, ULONG ttl)

{

	std::vector< BYTE >	byteArray;

	const void		*	byteArrayPtr	= NULL;

	DNSServiceErrorType	err				= 0;

	HRESULT				hr				= 0;

	BOOL				ok;



	// Convert the VARIANT

	ok = VariantToByteArray( &rdata, byteArray );

	require_action( ok, exit, err = kDNSServiceErr_Unknown );



	err = DNSServiceUpdateRecord( m_serviceObject->GetSubordRef(), m_rref, flags, ( uint16_t ) byteArray.size(), byteArray.size() > 0 ? &byteArray[ 0 ] : NULL, ttl );

	require_noerr( err, exit );



exit:



	return err;

}





STDMETHODIMP CDNSSDRecord::Remove(DNSSDFlags flags)

{

	DNSServiceErrorType	err = 0;



	err = DNSServiceRemoveRecord( m_serviceObject->GetSubordRef(), m_rref, flags );

	require_noerr( err, exit );



exit:



	return err;

}



