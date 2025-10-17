/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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
#include <Security/SecTrustedApplicationPriv.h>
#include <security_keychain/TrustedApplication.h>
#include <security_keychain/Certificate.h>
#include <securityd_client/ssclient.h>		// for code equivalence SPIs

#include "SecBridge.h"



#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
static inline CssmData cfData(CFDataRef data)
{
    return CssmData(const_cast<UInt8 *>(CFDataGetBytePtr(data)),
        CFDataGetLength(data));
}
#pragma clang diagnostic pop


CFTypeID
SecTrustedApplicationGetTypeID(void)
{
	BEGIN_SECAPI
	return gTypes().TrustedApplication.typeID;

	END_SECAPI1(_kCFRuntimeNotATypeID)
}


OSStatus
SecTrustedApplicationCreateFromPath(const char *path, SecTrustedApplicationRef *appRef)
{
	BEGIN_SECAPI
	SecPointer<TrustedApplication> app =
		path ? new TrustedApplication(path) : new TrustedApplication;
	Required(appRef) = app->handle();
	END_SECAPI
}

OSStatus SecTrustedApplicationCopyData(SecTrustedApplicationRef appRef,
	CFDataRef *dataRef)
{
	BEGIN_SECAPI
	const char *path = TrustedApplication::required(appRef)->path();
	Required(dataRef) = CFDataCreate(NULL, (const UInt8 *)path, strlen(path) + 1);
	END_SECAPI
}

OSStatus SecTrustedApplicationSetData(SecTrustedApplicationRef appRef,
	CFDataRef dataRef)
{
	BEGIN_SECAPI
	if (!dataRef)
		return errSecParam;
	TrustedApplication::required(appRef)->data(dataRef);
	END_SECAPI
}


OSStatus
SecTrustedApplicationValidateWithPath(SecTrustedApplicationRef appRef, const char *path)
{
	BEGIN_SECAPI
	TrustedApplication &app = *TrustedApplication::required(appRef);
	if (!app.verifyToDisk(path))
		return CSSMERR_CSP_VERIFY_FAILED;
	END_SECAPI
}


//
// Convert from/to external data representation
//
OSStatus SecTrustedApplicationCopyExternalRepresentation(
	SecTrustedApplicationRef appRef,
	CFDataRef *externalRef)
{
	BEGIN_SECAPI
	TrustedApplication &app = *TrustedApplication::required(appRef);
	Required(externalRef) = app.externalForm();
	END_SECAPI
}

OSStatus SecTrustedApplicationCreateWithExternalRepresentation(
	CFDataRef externalRef,
	SecTrustedApplicationRef *appRef)
{
	BEGIN_SECAPI
	Required(appRef) = (new TrustedApplication(externalRef))->handle();
	END_SECAPI
}


OSStatus
SecTrustedApplicationMakeEquivalent(SecTrustedApplicationRef oldRef,
	SecTrustedApplicationRef newRef, UInt32 flags)
{
	BEGIN_SECAPI
    return errSecParam;
	END_SECAPI
}

OSStatus
SecTrustedApplicationRemoveEquivalence(SecTrustedApplicationRef appRef, UInt32 flags)
{
	BEGIN_SECAPI
    return errSecParam;
	END_SECAPI
}


/*
 * Check to see if an application at a given path is a candidate for
 * pre-emptive code equivalency establishment
 */
OSStatus
SecTrustedApplicationIsUpdateCandidate(const char *installroot, const char *path)
{
    BEGIN_SECAPI
    return CSSMERR_DL_RECORD_NOT_FOUND;	// whatever
    END_SECAPI
}


/*
 * Point the system at another system root for equivalence use.
 * This is for system update installers (only)!
 */
OSStatus
SecTrustedApplicationUseAlternateSystem(const char *systemRoot)
{
	BEGIN_SECAPI
    return errSecParam;
	END_SECAPI
}


/*
 * Gateway between traditional SecTrustedApplicationRefs and the Code Signing
 * subsystem. Invisible to the naked eye, as of 10.5 (Leopard), these reference
 * may contain Cod e Signing Requirement objects (SecRequirementRefs). For backward
 * compatibility, these are handled implicitly at the SecAccess/SecACL layer.
 * However, Those Who Know can bridge the gap for additional functionality.
 */
OSStatus SecTrustedApplicationCreateFromRequirement(const char *description,
	SecRequirementRef requirement, SecTrustedApplicationRef *appRef)
{
	BEGIN_SECAPI
	if (description == NULL)
		description = "csreq://";	// default to "generic requirement"
	SecPointer<TrustedApplication> app = new TrustedApplication(description, requirement);
	Required(appRef) = app->handle();
	END_SECAPI
}

OSStatus SecTrustedApplicationCopyRequirement(SecTrustedApplicationRef appRef,
	SecRequirementRef *requirement)
{
	BEGIN_SECAPI
	Required(requirement) = TrustedApplication::required(appRef)->requirement();
	if (*requirement)
		CFRetain(*requirement);
	END_SECAPI
}


/*
 * Create an application group reference.
 */
OSStatus SecTrustedApplicationCreateApplicationGroup(const char *groupName,
	SecCertificateRef anchor, SecTrustedApplicationRef *appRef)
{
	BEGIN_SECAPI
	CFRef<SecRequirementRef> req;
	MacOSError::check(SecRequirementCreateGroup(CFTempString(groupName), anchor,
		kSecCSDefaultFlags, &req.aref()));
	string description = string("group://") + groupName;
	if (anchor) {
		Certificate *cert = Certificate::required(anchor);
		const CssmData &hash = cert->publicKeyHash();
		description = description + "?cert=" + cfString(cert->commonName())
			+ "&hash=" + hash.toHex();
	}
	SecPointer<TrustedApplication> app = new TrustedApplication(description, req);
	Required(appRef) = app->handle();

	END_SECAPI
}
