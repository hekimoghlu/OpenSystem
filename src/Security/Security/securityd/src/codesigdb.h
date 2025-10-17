/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 6, 2025.
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
// codesigdb - code-hash equivalence database
//
#ifndef _H_CODESIGDB
#define _H_CODESIGDB

#include "acls.h"
#include <security_cdsa_utilities/db++.h>
#include <security_cdsa_utilities/osxverifier.h>
#include <Security/CodeSigning.h>


class Process;
class CodeSignatures;


//
// A CodeSignatures object represents a database of code-signature equivalencies
// as (previously) expressed by a user and/or the system.
// You'll usually only need one of these.
//
class CodeSignatures {
public:
	//
	// Identity is an abstract class modeling a code-identity in the database.
	// It can represent either an existing or latent code-hash link.
	// Subclass must provide path and hash source functions.
	//
	class Identity {
		friend class CodeSignatures;
	public:
		Identity();
		virtual ~Identity();

		operator bool () const				{ return mState == valid; }
		std::string path()					{ return getPath(); }
		std::string name() 					{ return canonicalName(path()); }
		std::string trustedName() const		{ return mName; }

		static std::string canonicalName(const std::string &path);

		IFDUMP(void debugDump(const char *how = NULL) const);

		virtual std::string getPath() const = 0;
		virtual const CssmData getHash() const = 0;

	private:
		enum { untried, valid, invalid } mState;
		std::string mName;		// link db value (canonical name linked to)
	};

public:
	CodeSignatures();
	~CodeSignatures();

	void open(const char *path);

public:
	bool find(Identity &id, uid_t user);

	void makeLink(Identity &id, const std::string &ident, bool forUser = false, uid_t user = 0);

	void addLink(const CssmData &oldHash, const CssmData &newHash,
		const char *name, bool forSystem);
	void removeLink(const CssmData &hash, const char *name, bool forSystem);

	IFDUMP(void debugDump(const char *how = NULL) const);

public:
	bool verify(Process &process, const OSXVerifier &verifier, const AclValidationContext &context);

private:
	OSStatus matchSignedClientToLegacyACL(Process &process,
		const OSXVerifier &verifier, const AclValidationContext &context);

private:
	UnixPlusPlus::UnixDb mDb;

	// lock hierarchy: mUILock first, then mDatabaseLock, no back-off
	Mutex mDatabaseLock;			// controls mDb access
	Mutex mUILock;					// serializes user interaction
};



#endif //_H_CODESIGDB
