/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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
#ifndef _H_REQUIREMENTS
#define _H_REQUIREMENTS

#include "cs.h"
#include "requirement.h"

namespace Security {
namespace CodeSigning {


//
// A SecRequirement object acts as the API representation for a code
// requirement. All its semantics are within the Requirement object it holds.
// The SecRequirement just manages the API appearances.
//
class SecRequirement : public SecCFObject {
	NOCOPY(SecRequirement)
public:
	SECCFFUNCTIONS(SecRequirement, SecRequirementRef, errSecCSInvalidObjectRef, gCFObjects().Requirement)

	SecRequirement(const void *data, size_t length);
	SecRequirement(const Requirement *req, bool transferOwnership = false);
    virtual ~SecRequirement() _NOEXCEPT;
	
    bool equal(SecCFObject &other);
    CFHashCode hash();
	
	const Requirement *requirement() const { return mReq; }

private:
	const Requirement *mReq;
};


} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_REQUIREMENTS
