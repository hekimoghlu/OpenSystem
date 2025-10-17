/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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
// acl_keychain - a subject type for the protected-path
//		keychain prompt interaction model.
//
#ifndef _ACL_KEYCHAIN
#define _ACL_KEYCHAIN

#include <security_cdsa_utilities/cssmacl.h>
#include <string>


//
// This is the actual subject implementation class
//
class KeychainPromptAclSubject : public SimpleAclSubject {
	static const Version pumaVersion = 0;	// 10.0, 10.1 -> default selector (not stored)
	static const Version jaguarVersion = 1;	// 10.2 et al -> first version selector
	static const Version currentVersion = jaguarVersion; // what we write today
public:
    bool validates(const AclValidationContext &ctx) const;
    bool validates(const AclValidationContext &baseCtx, const TypedList &sample) const;
	bool validateExplicitly(const AclValidationContext &baseCtx, void (^always)()) const;
    CssmList toList(Allocator &alloc) const;
    bool hasAuthorizedForSystemKeychain() const;
    
    KeychainPromptAclSubject(string description, const CSSM_ACL_KEYCHAIN_PROMPT_SELECTOR &selector);
    
    void exportBlob(Writer::Counter &pub, Writer::Counter &priv);
    void exportBlob(Writer &pub, Writer &priv);
	
	uint32_t selectorFlags() const			{ return selector.flags; }
	bool selectorFlag(uint32_t flag) const	{ return selectorFlags() & flag; }
	
	IFDUMP(void debugDump() const);
    virtual CFStringRef createACLDebugString() const;

    static uint32_t getPromptAttempts();
    void addPromptAttempt(); // Use this only if you're going to call validateExplicitly out of the normal call hierarchy

public:
    class Maker : public AclSubject::Maker {
		friend class KeychainPromptAclSubject;
    public:
    	Maker(uint32_t mode)
			: AclSubject::Maker(CSSM_ACL_SUBJECT_TYPE_KEYCHAIN_PROMPT) { defaultMode = mode; }
    	KeychainPromptAclSubject *make(const TypedList &list) const;
    	KeychainPromptAclSubject *make(Version version, Reader &pub, Reader &priv) const;
	
	private:
		static uint32_t defaultMode;
    };
    
private:
    static uint32_t promptsValidated;

	CSSM_ACL_KEYCHAIN_PROMPT_SELECTOR selector; // selector structure
    string description;				// description blob (string)
	
private:
	static CSSM_ACL_KEYCHAIN_PROMPT_SELECTOR defaultSelector;
};


#endif //_ACL_KEYCHAIN
