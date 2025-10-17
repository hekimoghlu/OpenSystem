/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
// Code - SecCode API objects
//
#ifndef _H_CODE
#define _H_CODE

#include "cs.h"
#include "Requirements.h"
#include <security_utilities/utilities.h>

namespace Security {
namespace CodeSigning {


class SecStaticCode;


//
// A SecCode object represents running code in the system. It must be subclassed
// to implement a particular notion of code.
//
class SecCode : public SecCFObject {
	NOCOPY(SecCode)
	friend class KernelCode;	// overrides identify() to set mStaticCode/mCDHash
public:
	SECCFFUNCTIONS(SecCode, SecCodeRef, errSecCSInvalidObjectRef, gCFObjects().Code)

	SecCode(SecCode *host);
    virtual ~SecCode() _NOEXCEPT;
	
    bool equal(SecCFObject &other);
    CFHashCode hash();
	
	SecCode *host() const;
	bool isRoot() const { return host() == NULL; }
	SecStaticCode *staticCode();	// cached. Result lives as long as this SecCode
	CFDataRef cdHash();
	
	SecCodeStatus status();				// dynamic status
	void status(SecCodeStatusOperation operation, CFDictionaryRef arguments);

	// primary virtual drivers. Caller owns the result
	virtual void identify();
	virtual SecCode *locateGuest(CFDictionaryRef attributes);
	virtual SecStaticCode *identifyGuest(SecCode *guest, CFDataRef *cdhash);
	
	void checkValidity(SecCSFlags flags);
	virtual SecCodeStatus getGuestStatus(SecCode *guest);
	virtual void changeGuestStatus(SecCode *guest, SecCodeStatusOperation operation, CFDictionaryRef arguments);
	virtual void guestMatchesLightweightCodeRequirement(SecCode *guest, const Requirement* lwcr);
	
public:
	// perform "autolocation" (root-based heuristic). Caller owns the result
	static SecCode *autoLocateGuest(CFDictionaryRef attributes, SecCSFlags flags);

private:
	SecPointer<SecCode> mHost;
	bool mIdentified;							// called identify(), mStaticCode & mCDHash are valid
	SecPointer<SecStaticCode> mStaticCode;		// (static) code origin
	CFRef<CFDataRef> mCDHash;					// (dynamic) CodeDirectory hash as per host
};


} // end namespace CodeSigning
} // end namespace Security

#endif // !_H_CODE
