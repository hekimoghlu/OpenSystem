/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 29, 2022.
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
	@header SecTrustedApplicationPriv
	Not (yet?) public functions related to SecTrustedApplicationRef objects
*/

#ifndef _SECURITY_SECTRUSTEDAPPLICATIONPRIV_H_
#define _SECURITY_SECTRUSTEDAPPLICATIONPRIV_H_

#include <Security/SecTrustedApplication.h>
#include <Security/SecRequirementPriv.h>


#if defined(__cplusplus)
extern "C" {
#endif


/*
 * Determine whether the application at path satisfies the trust expressed in appRef.
 */
OSStatus
SecTrustedApplicationValidateWithPath(SecTrustedApplicationRef appRef, const char *path)
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);

/*!
	@function SecTrustedApplicationCreateFromRequirement
	@abstract Creates a trusted application reference based on an application
		URI description and a SecRequirementRef describing how it should be verified.
	@param description A URI-formatted string describing the intended meaning of
		the requirement being provided. This is for information purposes only
		and does not affect any actual validation being performed as a result.
		It may affect how the SecTrustedApplication is displayed or edited.
		If NULL, a default generic description is used.
	@param requirement A SecRequirementRef indicating what conditions an application
		must satisfy to be considered a match for this SecTrustedApplicationRef.
	@param app On return, contains a SecTrustedApplicationRef representing any
		code that satisfies the requirement argument.
	@result A result code. See SecBase.h and CSCommon.h.
*/
OSStatus SecTrustedApplicationCreateFromRequirement(const char *description,
	SecRequirementRef requirement, SecTrustedApplicationRef *app)
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);

/*!
	@function SecTrustedApplicationCopyRequirement
	@abstract If a SecTrustedApplicationRef contains a Code Signing requirement,
		it can be retrieved with this call. If no requirement is recorded, requirement
		is set to NULL and the call succeeds.
	@param appRef A trusted application reference to retrieve data from
	@param requirement Receives the SecRequirementRef contained in appRef, if any.
		If no Code Signing requirement is contained in appRef, *requirement is set
		to NULL and the call succeeds. This can happen if appRef was created from
		an unsigned application, or from sources that do not record code signing
		information such as keychain items made in version 10.4 or earlier of the
		system.
	@result A result code. See SecBase.h and CSCommon.h. It is not an error if
		no SecRequirementRef could be obtained.
 */
OSStatus SecTrustedApplicationCopyRequirement(SecTrustedApplicationRef appRef,
	SecRequirementRef *requirement)
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);


/*!
	@function SecTrustedApplicationCreateApplicationGroup
	@abstract Create a SecTrustedApplication object that represents an application
		group reference. It will match any application that has been marked as
		a member of the named group and was signed by a particular authority (anchor).
		Note that application groups are open-ended and more applications can be
		signed as members (by holders of suitable signing authorities) at any time.
		There is no way to reliably enumerate all members of an application group.
	@param groupName The name of the application group. If you define your own
		application group, use reverse domain notation (com.yourapp.yourgroup).
	@param anchor The anchor certificate that is required to seal the group.
		An application will be recognized as a member of the group only if it
		was signed with an identity that draws to this group. If NULL, requires
		signing by Apple.
	@param app On return, contains a SecTrustedApplicationRef representing any
		code that has been signed and marked as a member of the named application
		group.
	@result A result code. See SecBase.h and CSCommon.h.
 */
OSStatus SecTrustedApplicationCreateApplicationGroup(const char *groupName,
	SecCertificateRef anchor, SecTrustedApplicationRef *app)
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);


/*!
	@function SecTrustedApplicationCopyExternalRepresentation
	@abstract Create a pure-data form of a SecTrustedApplicationRef object suitable
		for persistent storage anywhere. This data can later be fed to
		SecTrustedApplicationCreateWithExternalRepresentation to create an equivalent
		SecTrustedApplicationRef. The data is variable size, and should be considered
		entirely opaque; its internal form is subject to change.
	@param appRef A valid SecTrustedApplicationRef of any kind.
	@param externalRef Upon successful return, contains a CFDataRef that can be
		stored as required.
	@result A result code. See SecBase.h and CSCommon.h.
 */
OSStatus SecTrustedApplicationCopyExternalRepresentation(
	SecTrustedApplicationRef appRef,
	CFDataRef *externalRef)
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);

/*!
	@function SecTrustedApplicationCreateWithExternalRepresentation
	@abstract Create a SecTrustedApplicationRef from an external data representation
		that was originally obtained with a call to SecTrustedApplicationCopyExternalRepresentation.
	@param externalRef A CFDataRef containing data produced by
		SecTrustedApplicationCopyExternalRepresentation. If this data was not obtained
		from that function, the behavior is undefined.
	@param appRef Upon successful return, a SecTrustedApplicationRef that is functionally
		equivalent to the original one used to obtain externalRef.
	@result A result code. See SecBase.h and CSCommon.h.
 */
OSStatus SecTrustedApplicationCreateWithExternalRepresentation(
	CFDataRef externalRef,
	SecTrustedApplicationRef *appRef)
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);


/*
 * Administrative editing of the system's application equivalence database
 */
enum {
	kSecApplicationFlagSystemwide =			0x1,
	kSecApplicationValidFlags =				kSecApplicationFlagSystemwide
};

OSStatus
SecTrustedApplicationMakeEquivalent(SecTrustedApplicationRef oldRef,
	SecTrustedApplicationRef newRef, UInt32 flags)
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);

OSStatus
SecTrustedApplicationRemoveEquivalence(SecTrustedApplicationRef appRef, UInt32 flags)
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);


/*
 * Check to see if an application at a given path is a candidate for
 * pre-emptive code equivalency establishment
 */
OSStatus
SecTrustedApplicationIsUpdateCandidate(const char *installroot, const char *path)
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);


/*
 * Point the system at another system root for equivalence use.
 * This is for system update installers (only)!
 */
OSStatus
SecTrustedApplicationUseAlternateSystem(const char *systemRoot)
API_DEPRECATED("SecKeychain is deprecated", macos(10.2, 10.10))
API_UNAVAILABLE(ios, watchos, tvos, bridgeos, macCatalyst);


#if defined(__cplusplus)
}
#endif

#endif /* !_SECURITY_SECTRUSTEDAPPLICATIONPRIV_H_ */
