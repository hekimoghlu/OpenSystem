/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 13, 2024.
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
/*!
	@header SecRequirementPrivLWCR
	SecRequirementPrivLWCR is a private counter-part to SecRequirement. Its contents are not
	official API, and are subject to change without notice.
*/
#ifndef _H_SECREQUIREMENTPRIVLWCR
#define _H_SECREQUIREMENTPRIVLWCR

#include <Security/SecRequirement.h>
#include <Security/SecCertificate.h>


#ifdef __cplusplus
extern "C" {
#endif

/*!
	@function SecRequirementCreateWithLightweightCodeRequirementData
	Create a SecRequirement object based on DER encoded LightweightCodeRequirement Data.
 
	@param lwcr A CFDataRef containing the DER encoded LightweightCodeRequirement Data.
	@param flags Optional flags. (Not used)
	@param result Upon success a SecRequirementRef for the requirement.
	@param errors An optional pointer to a CFErrorRef variable. If the call fails and something
	other than errSecSuccess is returned, then this argument is non-NULL and contains more
	information on the error. The caller must CFRelease() this error object when done.
	@result Upoon success, errSecSucces. Upon error, an OSStatus value documented in
	CSCommon.h or other headers.
*/
OSStatus SecRequirementCreateWithLightweightCodeRequirementData(CFDataRef lwcr, SecCSFlags flags,
																SecRequirementRef *result, CFErrorRef *errors);
#ifdef __cplusplus
}
#endif

#endif //_H_SECREQUIREMENTPRIV
