/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 21, 2022.
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
// osxsigner - MacOS X's standard code signing algorithm.
//
#ifndef _H_OSXVERIFIER
#define _H_OSXVERIFIER

#include <security_utilities/hashing.h>
#include <security_utilities/osxcode.h>
#include <security_utilities/blob.h>
#include <security_utilities/debugging_internal.h>
#include <security_cdsa_utilities/cssmdata.h>
#include <Security/CodeSigning.h>
#include <string>
#include <map>

#define LEGACY_HASH_LIMIT 16*1024

namespace Security {


//
// A standard OS X style signature verifier.
// This encapsulates the different modes of signing/verifying currently
// supported. It knows nothing about the way this is represented in
// keychain access control lists; this knowledge resides exclusively
// in acl_codesigning.
//
class OSXVerifier {
public:
	static const size_t legacyHashLimit = LEGACY_HASH_LIMIT;
	static const uint32_t commentAlignment = 4;
	
public:
	// make a Verifier from a code reference object
	OSXVerifier(OSXCode *code);		// makes both legacy hash and SecRequirement
	OSXVerifier(const SHA1::Byte *hash, const std::string &path); // just hash
	~OSXVerifier();

	// components
	const unsigned char *legacyHash() const { return mLegacyHash; }
	const std::string& path() const { return mPath; }
	SecRequirementRef requirement() const { return mRequirement; }

public:
	// handle other (not explicitly understood) information in the verifier
	class AuxMap : public std::map<BlobCore::Magic, BlobCore *> {
	public:
		AuxMap() { }
		AuxMap(const AuxMap &src);
		~AuxMap();
	};

	AuxMap::const_iterator beginAux() const { return mAuxiliary.begin(); }
	AuxMap::const_iterator endAux() const { return mAuxiliary.end(); }
	
	void add(const BlobCore *info);
	const BlobCore *find(BlobCore::Magic magic);

	template <class BlobType>
	static const BlobType *find()
	{ return static_cast<BlobType *>(find(BlobType::typeMagic)); }
	
public:
	static void makeLegacyHash(OSXCode *code, SHA1::Digest digest);

	IFDUMP(void dump() const);
	
private:
	SHA1::Digest mLegacyHash;		// legacy page hash
	std::string mPath;				// path to originating code (comment)
	CFCopyRef<SecRequirementRef> mRequirement; // CS-style requirement
	AuxMap mAuxiliary;				// other data (does not include mRequirement)
};

} // end namespace Security


#endif //_H_OSXVERIFIER
