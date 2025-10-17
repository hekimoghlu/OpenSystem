/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 21, 2022.
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
#ifndef	_SECURITY_SEC_IMPORT_EXPORT_PEM_H_
#define _SECURITY_SEC_IMPORT_EXPORT_PEM_H_

#include <Security/SecImportExport.h>
#include "SecExternalRep.h"

/* take these PEM header strings right from the authoritative source */
#include <openssl/pem.h>

/* Other PEM Header strings not defined in openssl */
#define PEM_STRING_DH_PUBLIC	"DH PUBLIC KEY"
#define PEM_STRING_DH_PRIVATE	"DH PRIVATE KEY"
#define PEM_STRING_PKCS12		"PKCS12"
#define PEM_STRING_SESSION		"SYMMETRIC KEY"
//#define PEM_STRING_ECDSA_PUBLIC	"EC PUBLIC KEY"
#define PEM_STRING_ECDSA_PRIVATE "EC PRIVATE KEY"

#ifdef	__cplusplus
extern "C" {
#endif

/*
 * PEM decode incoming data, appending SecImportRep's to specified array.
 * Returned SecImportReps may or may not have a known type and format. 
 * IF incoming data is not PEM or base64, we return errSecSuccess with *isPem false.
 */
OSStatus impExpParsePemToImportRefs(
	CFDataRef			importedData,
	CFMutableArrayRef	importReps,		// output appended here
	bool				*isPem);		// true means we think it was PEM regardless of 
										// final return code	

/*
 * PEM encode a single SecExportRep's data, appending to a CFData.
 */
OSStatus impExpPemEncodeExportRep(
	CFDataRef			derData,
	const char			*pemHeader,
	CFArrayRef			pemParamLines,  // optional 
	CFMutableDataRef	outData);
	
#ifdef	__cplusplus
}
#endif

#endif	/* _SECURITY_SEC_IMPORT_EXPORT_PEM_H_ */
