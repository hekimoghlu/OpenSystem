/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 19, 2024.
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
// acl_prompted - password-based validation with out-of-band prompting.
//
// This implements simple password-based subject types with out-of-band
// prompting (via SecurityAgent), somewhat as per the CSSM standard.
//
#ifndef _ACL_PROMPTED
#define _ACL_PROMPTED

#include <security_cdsa_utilities/acl_secret.h>


namespace Security {


//
// A PromptedAclSubject obtains its sample by prompting the user interactively
// through some prompting mechanism defined in the environment.
//
class PromptedAclSubject : public SecretAclSubject {
public:
    CssmList toList(Allocator &alloc) const;
    
    PromptedAclSubject(Allocator &alloc,
		const CssmData &prompt, const CssmData &password);
    PromptedAclSubject(Allocator &alloc,
		CssmManagedData &prompt, CssmManagedData &password);
	PromptedAclSubject(Allocator &alloc, const CssmData &prompt, bool cache = false);
    
    void exportBlob(Writer::Counter &pub, Writer::Counter &priv);
    void exportBlob(Writer &pub, Writer &priv);
	
	IFDUMP(void debugDump() const);
    virtual CFStringRef createACLDebugString() const;

public:
	class Environment : virtual public AclValidationEnvironment {
	public:
		virtual bool getSecret(CssmOwnedData &secret,
			const CssmData &prompt) const = 0;
	};

public:
    class Maker : public SecretAclSubject::Maker {
    public:
    	Maker() : SecretAclSubject::Maker(CSSM_ACL_SUBJECT_TYPE_PROMPTED_PASSWORD) { }
    	PromptedAclSubject *make(const TypedList &list) const;
    	PromptedAclSubject *make(Version, Reader &pub, Reader &priv) const;
    };
	
protected:
	bool getSecret(const AclValidationContext &context,
		const TypedList &subject, CssmOwnedData &secret) const;

private:
	CssmAutoData mPrompt;		// transparently handled prompt data
};

} // end namespace Security


#endif //_ACL_PROMPTED
