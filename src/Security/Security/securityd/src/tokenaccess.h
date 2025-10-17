/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 15, 2022.
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
// tokenaccess - access management to a TokenDatabase's Token's TokenDaemon's tokend
//
#ifndef _H_TOKENACCESS
#define _H_TOKENACCESS

#include "tokendatabase.h"
#include "tokenkey.h"
#include "server.h"


//
// Turn a Key into a TokenKey, when we know that it's that
//
inline TokenKey &myKey(Key &key)
{
	return safer_cast<TokenKey &>(key);
}


//
// The common access/retry/management framework for calls that go to the actual daemon.
//
class Access : public Token::Access {
public:
	Access(Token &token) : Token::Access(token), mIteration(0)
	{ Server::active().longTermActivity(); }
	template <class Whatever>
	Access(Token &token, Whatever &it) : Token::Access(token)
	{ add(it); Server::active().longTermActivity(); }
	
	void operator () (const CssmError &err);
	using Token::Access::operator ();

	void add(TokenAcl &acl)		{ mAcls.insert(&acl); }
	void add(TokenAcl *acl)		{ if (acl) mAcls.insert(acl); }
	void add(AclSource &src)	{ add(dynamic_cast<TokenAcl&>(src.acl())); }
	void add(AclSource *src)	{ if (src) add(*src); }
	void add(Key &key)			{ mAcls.insert(&myKey(key)); }

private:
	set<TokenAcl *> mAcls;		// TokenAcl subclasses to clear on retry
	unsigned int mIteration;	// iteration count (try, retry, give up)
};


//
// A nice little macro bracket to apply it.
// You must declare an Access called 'access' before doing
//	TRY
//		some actions
//		GUARD(a call to tokend)
//	DONE
//
#define TRY			for (;;) {
#define GUARD		try {
#define DONE		return; \
	} catch (const CssmError &error) { \
		access(error); \
	} }


#endif //_H_TOKENACCESS
