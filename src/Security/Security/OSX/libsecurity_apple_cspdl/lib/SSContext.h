/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 10, 2022.
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
// SSContext.h - Security Server contexts 
//
#ifndef _H_SS_CONTEXT
#define _H_SS_CONTEXT

#include <security_cdsa_plugin/CSPsession.h>
#include <securityd_client/ssclient.h>
#include <security_cdsa_utilities/digestobject.h>
#include <security_cdsa_client/cspclient.h>

//
// Parent class for all CSPContexts implemented in this CSP.  Currently the
// only thing we add is a reference to our creator's session.
//
class SSCSPSession;
class SSKey;

class SSContext : public CSPFullPluginSession::CSPContext
{
public:
	SSContext(SSCSPSession &session);
	~SSContext() { clearOutBuf(); }
	virtual void init(const Context &context, bool encoding);

protected:
	SecurityServer::ClientSession &clientSession();
	SSCSPSession &mSession;
	
	// mOutBuf provides a holding tank for implied final() operations
	// resulting from an outputSize(true, 0). This form of outputSize()
	// is understood to only occur just prior to the final() call. To avoid
	// an extra RPC (just to perform the outputSize(), most subclasses of
	// SSContext actually perform the final() operation at this time,
	// storing the result in mOutBuf. At final(), mOutBuf() is just copied
	// to the caller's supplied output buffer. 
	CssmData mOutBuf;		
	
	// We remember a pointer to the passed in context and assume it will
	// remain a valid from init(), update() all the way though the call to
	// final().
	const Context *mContext;
	
	void clearOutBuf();
	void copyOutBuf(CssmData &out);
};

// context for signature (sign and verify)
class SSSignatureContext : public SSContext
{
public:
	SSSignatureContext(SSCSPSession &session);
	~SSSignatureContext();
	virtual void init(const Context &context, bool signing);
	virtual void update(const CssmData &data);
	virtual size_t outputSize(bool final, size_t inSize);
	
	/* sign */
	void sign(CssmData &sig);
	virtual void final(CssmData &out);
	
	/* verify */
	virtual void final(const CssmData &in);
	
	/* for raw sign/verify - optionally called after init */ 
	virtual void setDigestAlgorithm(CSSM_ALGORITHMS digestAlg);

private:
	/* stash the context's key for final sign/verify */
	SecurityServer::KeyHandle mKeyHandle;	
	
	/* alg-dependent, calculated at init time */
	CSSM_ALGORITHMS	mSigAlg;		// raw signature alg
	CSSM_ALGORITHMS mDigestAlg;		// digest

	/* exactly one of these is used to collect updates */
	NullDigest 			*mNullDigest;
	CssmClient::Digest 	*mDigest;
};

// Context for GenerateRandom operations
class SSRandomContext : public SSContext
{
public:
	SSRandomContext(SSCSPSession &session);
	virtual void init(const Context &context, bool);
	virtual size_t outputSize(bool final, size_t inSize);
	virtual void final(CssmData &out);
	
private:
	uint32 mOutSize;		// spec'd in context at init() time 
};

// Context for Encrypt and Decrypt operations
class SSCryptContext : public SSContext
{
public:
	SSCryptContext(SSCSPSession &session);
	~SSCryptContext();
	virtual void init(const Context &context, bool encoding);
	virtual size_t inputSize(size_t outSize);
	virtual size_t outputSize(bool final, size_t inSize);
	virtual void minimumProgress(size_t &in, size_t &out);
	virtual void update(void *inp, size_t &inSize, void *outp,
						size_t &outSize);
	virtual void final(CssmData &out);

private:
	SecurityServer::KeyHandle mKeyHandle;
	NullDigest mNullDigest;						// accumulator
};

// Digest, using raw CSP
class SSDigestContext : public SSContext
{
public:
	SSDigestContext(SSCSPSession &session);
	~SSDigestContext();
	virtual void init(const Context &context, bool);
	virtual void update(const CssmData &data);
	virtual void final(CssmData &out);
	virtual size_t outputSize(bool final, size_t inSize);

private:
	CssmClient::Digest *mDigest;
};

// common class for MAC generate, verify
class SSMACContext : public SSContext
{
public:
	SSMACContext(SSCSPSession &session);
	virtual void init(const Context &context, bool);
	virtual void update(const CssmData &data);
	virtual size_t outputSize(bool final, size_t inSize);
	
	/* sign */
	void genMac(CssmData &mac);
	virtual void final(CssmData &out);
	/* verify */
	virtual void final(const CssmData &in);
	
private:
	SecurityServer::KeyHandle mKeyHandle;
	NullDigest mNullDigest;					// accumulator
};


#endif // _H_SS_CONTEXT
