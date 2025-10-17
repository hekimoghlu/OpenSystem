/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 17, 2022.
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
#ifndef	_SECURITY_SEC_IMPORT_EXPORT_AGG_H_
#define _SECURITY_SEC_IMPORT_EXPORT_AGG_H_

#include <Security/SecImportExport.h>
#include "SecExternalRep.h"

#ifdef	__cplusplus
extern "C" {
#endif

OSStatus impExpPkcs12Export(
	CFArrayRef							exportReps,		// SecExportReps
	SecItemImportExportFlags			flags,		
	const SecKeyImportExportParameters	*keyParams,		// optional 
	CFMutableDataRef					outData);		// output appended here

OSStatus impExpPkcs7Export(
	CFArrayRef							exportReps,		// SecExportReps
	SecItemImportExportFlags			flags,			// kSecItemPemArmour, etc. 	
	const SecKeyImportExportParameters	*keyParams,		// optional 
	CFMutableDataRef					outData);		// output appended here

OSStatus impExpPkcs12Import(
	CFDataRef							inData,
	SecItemImportExportFlags			flags,		
	const SecKeyImportExportParameters	*keyParams,		// optional 
	ImpPrivKeyImportState				&keyImportState,// IN/OUT
	/* caller must supply one of these */
	SecKeychainRef						importKeychain, // optional
	CSSM_CSP_HANDLE						cspHand,		// optional
	CFMutableArrayRef					outArray);		// optional, append here 
	
OSStatus impExpPkcs7Import(
	CFDataRef							inData,
	SecItemImportExportFlags			flags,		
	const SecKeyImportExportParameters	*keyParams,		// optional 
	SecKeychainRef						importKeychain,	// optional
	CFMutableArrayRef					outArray);		// optional, append here 

OSStatus impExpNetscapeCertImport(
	CFDataRef							inData,
	SecItemImportExportFlags			flags,		
	const SecKeyImportExportParameters	*keyParams,		// optional 
	SecKeychainRef						importKeychain, // optional
	CFMutableArrayRef					outArray);		// optional, append here 

OSStatus impExpImportCertCommon(
	const CSSM_DATA		*cdata,
	SecKeychainRef		importKeychain, // optional
	CFMutableArrayRef	outArray);		// optional, append here 

#ifdef	__cplusplus
}
#endif

#endif  /* _SECURITY_SEC_IMPORT_EXPORT_AGG_H_ */
