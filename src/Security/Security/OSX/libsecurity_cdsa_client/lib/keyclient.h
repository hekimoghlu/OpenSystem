/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 6, 2023.
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
// keyclient 
//
#ifndef _H_CDSA_CLIENT_KEYCLIENT
#define _H_CDSA_CLIENT_KEYCLIENT  1

#include <security_cdsa_client/aclclient.h>
#include <security_cdsa_client/cspclient.h>

namespace Security
{

namespace CssmClient
{

//
// Key
//
class KeyImpl : public ObjectImpl, public AclBearer, public CssmKey
{
public:
	KeyImpl(const CSP &csp);
	KeyImpl(const CSP &csp, const CSSM_KEY &key, bool copy = false);
	KeyImpl(const CSP &csp, const CSSM_DATA &keyData);
	virtual ~KeyImpl();
	
	CSP csp() const { return parent<CSP>(); }
	void deleteKey(const CSSM_ACCESS_CREDENTIALS *cred);
    
    CssmKeySize sizeInBits() const;

	// Acl manipulation
	void getAcl(AutoAclEntryInfoList &aclInfos, const char *selectionTag = NULL) const;
	void changeAcl(const CSSM_ACL_EDIT &aclEdit,
		const CSSM_ACCESS_CREDENTIALS *accessCred);

	// Acl owner manipulation
	void getOwner(AutoAclOwnerPrototype &owner) const;
	void changeOwner(const CSSM_ACL_OWNER_PROTOTYPE &newOwner,
		const CSSM_ACCESS_CREDENTIALS *accessCred = NULL);

	// Call this after completing the CSSM API call after having called Key::makeNewKey()
	void activate();

protected:
	void deactivate(); 
};

class Key : public Object
{
public:
	typedef KeyImpl Impl;
	explicit Key(Impl *impl) : Object(impl) {}
	
	Key() : Object(NULL) {}
	Key(const CSP &csp, const CSSM_KEY &key, bool copy = false)	: Object(new Impl(csp, key, copy)) {}
	Key(const CSP &csp, const CSSM_DATA &keyData)	: Object(new Impl(csp, keyData)) {}

	// Creates an inactive key, client must call activate() after this.
	Key(const CSP &csp) : Object(new Impl(csp)) {}

	Impl *operator ->() const			{ return (*this) ? &impl<Impl>() : NULL; }
	Impl &operator *() const			{ return impl<Impl>(); }

	// Conversion operators to CssmKey baseclass.
	operator const CssmKey * () const	{ return (*this) ? &(**this) : NULL; }
	operator const CssmKey & () const	{ return **this; }
	
	// a few shortcuts to make life easier
	CssmKey::Header &header() const		{ return (*this)->header(); }

	// Creates an inactive key, client must call activate() after this.
	CssmKey *makeNewKey(const CSP &csp)	{ (*this) = Key(csp); return &(**this); }
    
    // inquiries
    CssmKeySize sizeInBits() const		{ return (*this)->sizeInBits(); }
};


struct KeySpec {
	CSSM_KEYUSE usage;
	CSSM_KEYATTR_FLAGS attributes;
	const CssmData *label;
	//add rc context
	
	KeySpec(CSSM_KEYUSE u, CSSM_KEYATTR_FLAGS a) : usage(u), attributes(a), label(NULL) { }
	KeySpec(CSSM_KEYUSE u, CSSM_KEYATTR_FLAGS a, const CssmData &l) : usage(u), attributes(a), label(&l) { }
};

} // end namespace CssmClient

} // end namespace Security


#endif // _H_CDSA_CLIENT_KEYCLIENT
