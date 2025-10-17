/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 26, 2025.
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
// Requirements - SecRequirement API objects
//
#include "Requirements.h"

namespace Security {
namespace CodeSigning {


//
// Create from a Requirement blob in memory
//
SecRequirement::SecRequirement(const void *data, size_t length)
	: mReq(NULL)
{
	const Requirement *req = (const Requirement *)data;
	if (!req->validateBlob(length))
		MacOSError::throwMe(errSecCSReqInvalid);
	mReq = req->clone();
}


//
// Create from a genuine Requirement object
//
SecRequirement::SecRequirement(const Requirement *req, bool transferOwnership)
	: mReq(NULL)
{
	if (!req->validateBlob())
		MacOSError::throwMe(errSecCSReqInvalid);
	
	if (transferOwnership)
		mReq = req;
	else
		mReq = req->clone();
} 

//
// Clean up a SecRequirement object
//
SecRequirement::~SecRequirement() _NOEXCEPT
try {
	::free((void *)mReq);
} catch (...) {
	return;
}


//
// CF-level comparison of SecRequirement objects compares the entire requirement
// structure for equality. This means that two requirement programs are recognized
// as equal if they're written identically (modulo comments and syntactic sugar).
// Obviously, equality of outcome is not in the cards. :-)
//
bool SecRequirement::equal(SecCFObject &secOther)
{
	SecRequirement *other = static_cast<SecRequirement *>(&secOther);
	return !memcmp(this->requirement(), other->requirement(), this->requirement()->length());
}

CFHashCode SecRequirement::hash()
{
	return CFHash(CFTempDataWrap(*this->requirement()));
}


} // end namespace CodeSigning
} // end namespace Security
