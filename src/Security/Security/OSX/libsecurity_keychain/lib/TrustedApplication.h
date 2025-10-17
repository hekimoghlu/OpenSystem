/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 14, 2022.
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
// TrustedApplication.h - TrustedApplication control wrappers
//
#ifndef _SECURITY_TRUSTEDAPPLICATION_H_
#define _SECURITY_TRUSTEDAPPLICATION_H_

#include <Security/SecTrustedApplication.h>
#include <security_cdsa_utilities/cssmdata.h>
#include <security_cdsa_utilities/cssmaclpod.h>
#include <security_cdsa_utilities/acl_codesigning.h>
#include <security_utilities/seccfobject.h>
#include "SecCFTypes.h"


namespace Security {
namespace KeychainCore {


//
// TrustedApplication actually denotes a signed executable
// on disk as used by the ACL subsystem. Much useful
// information is encapsulated in the 'comment' field that
// is stored with the ACL subject. TrustedApplication does
// not interpret this value, leaving its meaning to its caller.
//
class TrustedApplication : public SecCFObject {
	NOCOPY(TrustedApplication)
public:
	SECCFFUNCTIONS(TrustedApplication, SecTrustedApplicationRef, errSecInvalidItemRef, gTypes().TrustedApplication)

	TrustedApplication(const TypedList &subject);	// from ACL subject form
	TrustedApplication(const std::string &path);	// from code on disk
	TrustedApplication();							// for current application
	TrustedApplication(const std::string &path, SecRequirementRef requirement); // with requirement and aux. path
	TrustedApplication(CFDataRef external);			// from external representation
	~TrustedApplication();

	const char *path() const { return mForm->path().c_str(); }
	CssmData legacyHash() const	{ return CssmData::wrap(mForm->legacyHash(), SHA1::digestLength); }
	SecRequirementRef requirement() const { return mForm->requirement(); }

	void data(CFDataRef data);
	CFDataRef externalForm() const;

	CssmList makeSubject(Allocator &allocator);

	bool verifyToDisk(const char *path);		// verify against on-disk image

private:
	RefPointer<CodeSignatureAclSubject> mForm;
};

} // end namespace KeychainCore
} // end namespace Security

#endif // !_SECURITY_TRUSTEDAPPLICATION_H_
