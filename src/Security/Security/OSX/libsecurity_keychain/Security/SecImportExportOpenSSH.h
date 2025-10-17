/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 1, 2022.
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
 * SecImportExportOpenSSH.h - support for importing and exporting OpenSSH keys. 
 *
 */

#ifndef	_SEC_IMPORT_EXPORT_OPENSSH_H_
#define _SEC_IMPORT_EXPORT_OPENSSH_H_

#include <Security/SecImportExport.h>
#include <security_cdsa_utilities/cssmdata.h>

#ifdef	__cplusplus
extern "C" {
#endif

/* 
 * Infer PrintName attribute from raw key's 'comment' field. 
 * Returned string is mallocd and must be freed by caller. 
 */
extern char *impExpOpensshInferPrintName(
	CFDataRef external, 
	SecExternalItemType externType, 
	SecExternalFormat externFormat);
	
/* 
 * Infer DescriptiveData (i.e., comment) from a SecKeyRef's PrintName
 * attribute.
 */
extern void impExpOpensshInferDescData(
	SecKeyRef keyRef,
	CssmOwnedData &descData);
	
/*
 * If cspHand is provided instead of importKeychain, the CSP 
 * handle MUST be for the CSPDL, not for the raw CSP.
 */
extern OSStatus impExpWrappedOpenSSHImport(
	CFDataRef							inData,
	SecKeychainRef						importKeychain, // optional
	CSSM_CSP_HANDLE						cspHand,		// required
	SecItemImportExportFlags			flags,
	const SecKeyImportExportParameters	*keyParams,		// optional 
	const char							*printName,
	CFMutableArrayRef					outArray);		// optional, append here 

extern OSStatus impExpWrappedOpenSSHExport(
	SecKeyRef							secKey,
	SecItemImportExportFlags			flags,		
	const SecKeyImportExportParameters	*keyParams,		// optional 
	const CssmData						&descData,
	CFMutableDataRef					outData);		// output appended here

#ifdef	__cplusplus
}
#endif

#endif	/* _SEC_IMPORT_EXPORT_OPENSSH_H_ */
