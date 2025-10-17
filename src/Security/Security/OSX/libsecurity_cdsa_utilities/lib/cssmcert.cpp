/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 22, 2022.
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
//
// cssmcert - CSSM layer certificate (CL) related objects.
//
#include <security_cdsa_utilities/cssmcert.h>
#include <security_utilities/debugging.h>


namespace Security {


//
// Construct an EncodedCertificate
//
EncodedCertificate::EncodedCertificate(CSSM_CERT_TYPE type, CSSM_CERT_ENCODING enc,
	const CSSM_DATA *data)
{
	clearPod();
	CertType = type;
	CertEncoding = enc;
	if (data)
		CertBlob = *data;
}


//
// Construct an empty CertGroup.
//
CertGroup::CertGroup(CSSM_CERT_TYPE ctype,
        CSSM_CERT_ENCODING encoding, CSSM_CERTGROUP_TYPE type)
{
    clearPod();
    CertType = ctype;
    CertEncoding = encoding;
    CertGroupType = type;
}


//
// Free all memory in a CertGroup
//
void CertGroup::destroy(Allocator &allocator)
{
	switch (type()) {
	case CSSM_CERTGROUP_DATA:
		// array of CSSM_DATA elements
		for (uint32 n = 0; n < count(); n++)
			allocator.free(blobCerts()[n].data());
		allocator.free (blobCerts ());
		break;
	case CSSM_CERTGROUP_ENCODED_CERT:
		for (uint32 n = 0; n < count(); n++)
			allocator.free(encodedCerts()[n].data());
		allocator.free (blobCerts ());
		break;
	case CSSM_CERTGROUP_PARSED_CERT:
		// CSSM_PARSED_CERTS array -- unimplemented
	case CSSM_CERTGROUP_CERT_PAIR:
		// CSSM_CERT_PAIR array -- unimplemented
		break;
	}
}


}	// end namespace Security
