/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 17, 2022.
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
/*
 * SecImportExportPkcs8.h - support for generating and parsing/decoding 
 * private keys in PKCS8 format.  
 */
 
#ifndef _SEC_IMPORT_EXPORT_PKCS8_H_
#define _SEC_IMPORT_EXPORT_PKCS8_H_

#include <Security/secasn1t.h>
#include <Security/keyTemplates.h>	

#ifdef  __cplusplus
extern "C" {
#endif

/*
 * If cspHand is provided instead of importKeychain, the CSP 
 * handle MUST be for the CSPDL, not for the raw CSP.
 */
OSStatus impExpPkcs8Import(
	CFDataRef							inData,
	SecKeychainRef						importKeychain, // optional
	CSSM_CSP_HANDLE						cspHand,		// required
	SecItemImportExportFlags			flags,
	const SecKeyImportExportParameters	*keyParams,		// optional 
	CFMutableArrayRef					outArray);		// optional, append here 

OSStatus impExpPkcs8Export(
	SecKeyRef							secKey,
	SecItemImportExportFlags			flags,		
	const SecKeyImportExportParameters	*keyParams,		// optional 
	CFMutableDataRef					outData,		// output appended here
	const char							**pemHeader);	

#ifdef  __cplusplus
}
#endif

#endif  /* _SEC_IMPORT_EXPORT_PKCS8_H_ */

