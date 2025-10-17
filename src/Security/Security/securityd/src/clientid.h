/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 19, 2022.
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
// clientid - track and manage identity of securityd clients
//
#ifndef _H_CLIENTID
#define _H_CLIENTID

#include "codesigdb.h"
#include <Security/SecCode.h>
#include <security_utilities/ccaudit.h>
#include <security_utilities/cfutilities.h>
#include <string>


//
// A ClientIdentification object is a mix-in class that tracks
// the identity of associated client processes and their sub-entities
// (aka Code Signing Guest objects).
//
class ClientIdentification : public CodeSignatures::Identity {
public:
	ClientIdentification();

	std::string partitionId() const;
	AclSubject* copyAclSubject() const;

	// CodeSignatures::Identity personality
	string getPath() const;
	const CssmData getHash() const;
	OSStatus checkValidity(SecCSFlags flags, SecRequirementRef requirement) const;
	OSStatus copySigningInfo(SecCSFlags flags, CFDictionaryRef *info) const;
    bool checkAppleSigned() const;
	bool hasEntitlement(const char *name) const;

protected:
	//
	// Access to the underlying SecCodeRef should only be made from methods of
	// this class, which must take the appropriate mutex when accessing them.
	//
	SecCodeRef processCode() const;
	SecCodeRef currentGuest() const;

	void setup(Security::CommonCriteria::AuditToken const &audit);

public:
	IFDUMP(void dump());

private:
	CFRef<SecCodeRef> mClientProcess;	// process-level client object

	mutable RecursiveMutex mValidityCheckLock; // protects validity check

	mutable Mutex mLock;				// protects everything below

	struct GuestState {
		GuestState() : gotHash(false) { }
		CFRef<SecCodeRef> code;
		mutable bool gotHash;
		mutable SHA1::Digest legacyHash;
		mutable dispatch_time_t lastTouchTime; // so we can eject the LRU entries
	};
	typedef std::map<SecGuestRef, GuestState> GuestMap;
	mutable GuestMap mGuests;
	const static size_t kMaxGuestMapSize = 20;

	mutable std::string mClientPartitionId;
	mutable bool mGotPartitionId;

	GuestState *current() const;
	static std::string partitionIdForProcess(SecStaticCodeRef code);
};


//
// Bonus function
//
std::string codePath(SecStaticCodeRef code);


#endif //_H_CLIENTID
