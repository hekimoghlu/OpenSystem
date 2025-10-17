/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 21, 2023.
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
#ifndef _H_TOKENACL
#define _H_TOKENACL


//
// tokenacl - Token-based ACL implementation
//
#include "acls.h"
#include <security_cdsa_utilities/acl_preauth.h>

class Token;
class TokenDatabase;


//
// The Token version of a SecurityServerAcl.
//
class TokenAcl : public virtual SecurityServerAcl {
public:
	TokenAcl();
	
	typedef unsigned int ResetGeneration;

public:
	// implement SecurityServerAcl
	void getOwner(AclOwnerPrototype &owner);
	void getAcl(const char *tag, uint32 &count, AclEntryInfo *&acls);
    void changeAcl(const AclEdit &edit, const AccessCredentials *cred,
		Database *relatedDatabase);
	void changeOwner(const AclOwnerPrototype &newOwner, const AccessCredentials *cred,
		Database *relatedDatabase);

	void instantiateAcl();
	void changedAcl();

public:
	// required from our MDC
	virtual Token &token() = 0;
	virtual GenericHandle tokenHandle() const = 0;
	
protected:
	void invalidateAcl()	{ mLastReset = 0; }
	void pinChange(unsigned int pin, CSSM_ACL_HANDLE handle, TokenDatabase &database);
	
private:
	ResetGeneration mLastReset;
};


#endif //_H_TOKENACL
