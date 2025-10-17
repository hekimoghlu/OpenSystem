/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 23, 2024.
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
// TrustedApplication.cpp
//
#include <security_keychain/TrustedApplication.h>
#include <security_keychain/ACL.h>
#include <security_utilities/osxcode.h>
#include <security_utilities/trackingallocator.h>
#include <security_cdsa_utilities/acl_codesigning.h>
#include <sys/syslimits.h>
#include <memory>

using namespace KeychainCore;


//
// Create a TrustedApplication from a code-signing ACL subject.
// Throws ACL::ParseError if the subject is unexpected.
//
TrustedApplication::TrustedApplication(const TypedList &subject)
{
	try {
		CodeSignatureAclSubject::Maker maker;
		mForm = maker.make(subject);
		secinfo("trustedapp", "%p created from list form", this);
		IFDUMPING("codesign", mForm->AclSubject::dump("STApp created from list"));
	} catch (...) {
		throw ACL::ParseError();
	}
}


//
// Create a TrustedApplication from a path-to-object-on-disk
//
TrustedApplication::TrustedApplication(const std::string &path)
{
	RefPointer<OSXCode> code(OSXCode::at(path));
	mForm = new CodeSignatureAclSubject(OSXVerifier(code));
	secinfo("trustedapp", "%p created from path %s", this, path.c_str());
	IFDUMPING("codesign", mForm->AclSubject::dump("STApp created from path"));
}


//
// Create a TrustedApplication for the calling process
//
TrustedApplication::TrustedApplication()
{
	//@@@@ should use CS's idea of "self"
	RefPointer<OSXCode> me(OSXCode::main());
	mForm = new CodeSignatureAclSubject(OSXVerifier(me));
	secinfo("trustedapp", "%p created from self", this);
	IFDUMPING("codesign", mForm->AclSubject::dump("STApp created from self"));
}


//
// Create a TrustedApplication from a SecRequirementRef.
// Note that the path argument is only stored for documentation;
// it is NOT used to denote anything on disk.
//
TrustedApplication::TrustedApplication(const std::string &path, SecRequirementRef reqRef)
{
	CFRef<CFDataRef> reqData;
	MacOSError::check(SecRequirementCopyData(reqRef, kSecCSDefaultFlags, &reqData.aref()));
	mForm = new CodeSignatureAclSubject(NULL, path);
	mForm->add((const BlobCore *)CFDataGetBytePtr(reqData));
	secinfo("trustedapp", "%p created from path %s and requirement %p",
		this, path.c_str(), reqRef);
	IFDUMPING("codesign", mForm->debugDump());
}


TrustedApplication::~TrustedApplication()
{ /* virtual */ }


//
// Convert from/to external data form.
//
// Since a TrustedApplication's data is essentially a CodeSignatureAclSubject,
// we just use the subject's externalizer to produce the data. That requires us
// to use the somewhat idiosyncratic linearizer used by CSSM ACL subjects, but
// that's a small price to pay for consistency.
//
TrustedApplication::TrustedApplication(CFDataRef external)
{
	AclSubject::Reader pubReader(CFDataGetBytePtr(external)), privReader;
	mForm = CodeSignatureAclSubject::Maker().make(0, pubReader, privReader);
}

CFDataRef TrustedApplication::externalForm() const
{
	AclSubject::Writer::Counter pubCounter, privCounter;
	mForm->exportBlob(pubCounter, privCounter);
	if (privCounter > 0)	// private exported data - format violation
		CssmError::throwMe(CSSMERR_CSSM_INTERNAL_ERROR);
	CFRef<CFMutableDataRef> data = CFDataCreateMutable(NULL, pubCounter);
	CFDataSetLength(data, pubCounter);
	if (CFDataGetLength(data) < CFIndex(pubCounter))
		CFError::throwMe();
	AclSubject::Writer pubWriter(CFDataGetMutableBytePtr(data)), privWriter;
	mForm->exportBlob(pubWriter, privWriter);
	return data.yield();
}

void TrustedApplication::data(CFDataRef data)
{
	const char *p = (const char *)CFDataGetBytePtr(data);
	const std::string path(p, p + CFDataGetLength(data));
	RefPointer<OSXCode> code(OSXCode::at(path));
	mForm = new CodeSignatureAclSubject(OSXVerifier(code));
}

//
// Direct verification interface.
// If path == NULL, we verify against the running code itself.
//
bool TrustedApplication::verifyToDisk(const char *path)
{
	if (SecRequirementRef requirement = mForm->requirement()) {
		secinfo("trustedapp", "%p validating requirement against path %s", this, path);
		CFRef<SecStaticCodeRef> ondisk;
		if (path)
			MacOSError::check(SecStaticCodeCreateWithPath(CFTempURL(path),
				kSecCSDefaultFlags, &ondisk.aref()));
		else
			MacOSError::check(SecCodeCopySelf(kSecCSDefaultFlags, (SecCodeRef *)&ondisk.aref()));
		return SecStaticCodeCheckValidity(ondisk, kSecCSDefaultFlags, requirement) == errSecSuccess;
	} else {
		secinfo("trustedapp", "%p validating hash against path %s", this, path);
		RefPointer<OSXCode> code = path ? OSXCode::at(path) : OSXCode::main();
		SHA1::Digest ondiskDigest;
		OSXVerifier::makeLegacyHash(code, ondiskDigest);
		return memcmp(ondiskDigest, mForm->legacyHash(), sizeof(ondiskDigest)) == 0;
	}
}


//
// Produce a TypedList representing a code-signing ACL subject
// for this application.
// Memory is allocated from the allocator given, and belongs to
// the caller.
//
CssmList TrustedApplication::makeSubject(Allocator &allocator)
{
	return mForm->toList(allocator);
}


