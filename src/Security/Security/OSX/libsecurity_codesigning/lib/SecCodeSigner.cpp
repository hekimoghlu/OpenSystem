/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 9, 2023.
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
// SecCode - API frame for SecCode objects.
//
// Note that some SecCode* functions take SecStaticCodeRef arguments in order to
// accept either static or dynamic code references, operating on the respective
// StaticCode. Those functions are in SecStaticCode.cpp, not here, despite their name.
//
#include "cs.h"
#include "CodeSigner.h"
#include "cskernel.h"

using namespace CodeSigning;


//
// Parameter keys
//
const CFStringRef kSecCodeSignerApplicationData = CFSTR("application-specific");
const CFStringRef kSecCodeSignerDetached =		CFSTR("detached");
const CFStringRef kSecCodeSignerDigestAlgorithm = CFSTR("digest-algorithm");
const CFStringRef kSecCodeSignerDryRun =		CFSTR("dryrun");
const CFStringRef kSecCodeSignerEntitlements =	CFSTR("entitlements");
const CFStringRef kSecCodeSignerFlags =			CFSTR("flags");
const CFStringRef kSecCodeSignerForceLibraryEntitlements = CFSTR("force-library-entitlements");
const CFStringRef kSecCodeSignerIdentifier =	CFSTR("identifier");
const CFStringRef kSecCodeSignerIdentifierPrefix = CFSTR("identifier-prefix");
const CFStringRef kSecCodeSignerIdentity =		CFSTR("signer");
const CFStringRef kSecCodeSignerPageSize =		CFSTR("pagesize");
const CFStringRef kSecCodeSignerRequirements =	CFSTR("requirements");
const CFStringRef kSecCodeSignerResourceRules =	CFSTR("resource-rules");
const CFStringRef kSecCodeSignerSDKRoot =		CFSTR("sdkroot");
const CFStringRef kSecCodeSignerSigningTime =	CFSTR("signing-time");
const CFStringRef kSecCodeSignerRequireTimestamp = CFSTR("timestamp-required");
const CFStringRef kSecCodeSignerTimestampServer = CFSTR("timestamp-url");
const CFStringRef kSecCodeSignerTimestampAuthentication = CFSTR("timestamp-authentication");
const CFStringRef kSecCodeSignerTimestampOmitCertificates =	CFSTR("timestamp-omit-certificates");
const CFStringRef kSecCodeSignerPreserveMetadata = CFSTR("preserve-metadata");
const CFStringRef kSecCodeSignerTeamIdentifier =	CFSTR("teamidentifier");
const CFStringRef kSecCodeSignerPlatformIdentifier = CFSTR("platform-identifier");
const CFStringRef kSecCodeSignerRuntimeVersion = CFSTR("runtime-version");
const CFStringRef kSecCodeSignerPreserveAFSC = 	CFSTR("preserve-afsc");
const CFStringRef kSecCodeSignerOmitAdhocFlag =	CFSTR("omit-adhoc-flag");

const CFStringRef kSecCodeSignerLaunchConstraintSelf = CFSTR("lwcr-self");
const CFStringRef kSecCodeSignerLaunchConstraintParent = CFSTR("lwcr-parent");
const CFStringRef kSecCodeSignerLaunchConstraintResponsible = CFSTR("lwcr-responsible");
const CFStringRef kSecCodeSignerLibraryConstraint = CFSTR("lwcr-library");

// Keys for signature editing
const CFStringRef kSecCodeSignerEditCpuType = 	CFSTR("edit-cpu-type");
const CFStringRef kSecCodeSignerEditCpuSubtype = CFSTR("edit-cpu-subtype");
const CFStringRef kSecCodeSignerEditCMS = 		CFSTR("edit-cms");



//
// CF-standard type code functions
//
CFTypeID SecCodeSignerGetTypeID(void)
{
	BEGIN_CSAPI
	return gCFObjects().CodeSigner.typeID;
    END_CSAPI1(_kCFRuntimeNotATypeID)
}


//
// Create a signer object
//
OSStatus SecCodeSignerCreate(CFDictionaryRef parameters, SecCSFlags flags,
	SecCodeSignerRef *signerRef)
{
	BEGIN_CSAPI
		
	checkFlags(flags,
		  kSecCSEditSignature
		| kSecCSRemoveSignature
		| kSecCSSignPreserveSignature
		| kSecCSSignNestedCode
		| kSecCSSignOpaque
		| kSecCSSignV1
		| kSecCSSignNoV1
		| kSecCSSignBundleRoot
		| kSecCSSignStrictPreflight
		| kSecCSSignGeneratePEH
		| kSecCSSignGenerateEntitlementDER
		| kSecCSStripDisallowedXattrs
		| kSecCSSingleThreadedSigning);
	SecPointer<SecCodeSigner> signer = new SecCodeSigner(flags);
	signer->parameters(parameters);
	CodeSigning::Required(signerRef) = signer->handle();

    END_CSAPI
}


//
// Generate a signature
//
OSStatus SecCodeSignerAddSignature(SecCodeSignerRef signerRef,
	SecStaticCodeRef codeRef, SecCSFlags flags)
{
	return SecCodeSignerAddSignatureWithErrors(signerRef, codeRef, flags, NULL);
}

OSStatus SecCodeSignerAddSignatureWithErrors(SecCodeSignerRef signerRef,
	SecStaticCodeRef codeRef, SecCSFlags flags, CFErrorRef *errors)
{
	BEGIN_CSAPI
	checkFlags(flags,
		kSecCSReportProgress
	);
	SecCodeSigner::required(signerRef)->sign(SecStaticCode::required(codeRef), flags);
    END_CSAPI_ERRORS
}
