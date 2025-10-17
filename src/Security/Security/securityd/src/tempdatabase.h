/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 29, 2021.
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
// tempdatabase - temporary (scratch) storage for keys
//
// A TempDatabase locally manages keys using the AppleCSP while providing
// no persistent storage. Keys live until they are no longer referenced in
// client space, at which point they are destroyed.
//
#ifndef _H_TEMPDATABASE
#define _H_TEMPDATABASE

#include "localdatabase.h"


//
// A TempDatabase is simply a container of (a subclass of) LocalKey.
// When it dies, all its contents irretrievably vanish. There is no DbCommon
// or global object; each TempDatabase is completely distinct.
// Database ACLs are not (currently) supported on TempDatabases.
//
class TempDatabase : public LocalDatabase {
public:
	TempDatabase(Process &proc);

	const char *dbName() const;
    uint32 dbVersion();
	bool transient() const;
	
	RefPointer<Key> makeKey(const CssmKey &newKey, uint32 moreAttributes,
		const AclEntryPrototype *owner);
	
	void generateKey(const Context &context,
		 const AccessCredentials *cred, 
		 const AclEntryPrototype *owner, uint32 usage, 
		 uint32 attrs, RefPointer<Key> &newKey);
	
protected:
	void getSecurePassphrase(const Context &context, string &passphrase);
	void makeSecurePassphraseKey(const Context &context, const AccessCredentials *cred, 
								 const AclEntryPrototype *owner, uint32 usage, 
								 uint32 attrs, RefPointer<Key> &newKey);
};

#endif //_H_TEMPDATABASE
