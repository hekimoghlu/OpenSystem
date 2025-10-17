/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 25, 2025.
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
#include "cs.h"
#include "CodeSignerRemote.h"
#include "cskernel.h"

using namespace CodeSigning;

CFTypeID
SecCodeSignerRemoteGetTypeID(void)
{
	BEGIN_CSAPI
	return gCFObjects().CodeSignerRemote.typeID;
	END_CSAPI1(_kCFRuntimeNotATypeID)
}

OSStatus
SecCodeSignerRemoteCreate(CFDictionaryRef parameters,
						  CFArrayRef signingCertificateChain,
						  SecCSFlags flags,
						  SecCodeSignerRemoteRef * __nonnull CF_RETURNS_RETAINED signerRef,
						  CFErrorRef *errors)
{
	BEGIN_CSAPI
	checkFlags(flags,
			   kSecCSSignPreserveSignature
			   | kSecCSSignV1
			   | kSecCSSignNoV1
			   | kSecCSSignBundleRoot
			   | kSecCSSignStrictPreflight
			   | kSecCSSignGeneratePEH
			   | kSecCSSignGenerateEntitlementDER);

	SecPointer<SecCodeSignerRemote> signer = new SecCodeSignerRemote(flags, signingCertificateChain);
	signer->parameters(parameters);
	CodeSigning::Required(signerRef) = signer->handle();
	END_CSAPI_ERRORS
}

OSStatus
SecCodeSignerRemoteAddSignature(SecCodeSignerRemoteRef signerRef,
								SecStaticCodeRef codeRef,
								SecCSFlags flags,
								SecCodeRemoteSignHandler signHandler,
								CFErrorRef *errors)
{
	BEGIN_CSAPI
	checkFlags(flags, 0);
	SecPointer<SecCodeSignerRemote> signer = SecCodeSignerRemote::required(signerRef);
	signer->sign(SecStaticCode::required(codeRef), flags, signHandler);
	END_CSAPI_ERRORS
}
