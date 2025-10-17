/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 15, 2025.
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
/*
 * MacContext.h - AppleCSPContext for HMAC{SHA1,MD5}
 */

#ifndef	_MAC_CONTEXT_H_
#define _MAC_CONTEXT_H_

#include <AppleCSPContext.h>
#include <CommonCrypto/CommonHMAC.h>

/* 
 * TLS Export Ciphers require HMAC calculation with a secret key
 * size of 0 bytes. We'd really like to enforce a minimum key size equal 
 * the digest size, per RFC 2104, but TLS precludes that.
 */
#define HMAC_MIN_KEY_SIZE		0
#define HMAC_SHA_MIN_KEY_SIZE	HMAC_MIN_KEY_SIZE
#define HMAC_MD5_MIN_KEY_SIZE	HMAC_MIN_KEY_SIZE
#define HMAC_MAX_KEY_SIZE		2048

class MacContext : public AppleCSPContext  {
public:
	MacContext(
		AppleCSPSession &session,
		CSSM_ALGORITHMS alg) : 
			AppleCSPContext(session), 
			mAlg(alg),
			mDigestSize(0) { }
	~MacContext();
	
	/* called out from CSPFullPluginSession....
	 * both generate and verify: */
	void init(const Context &context, bool isSigning);
	void update(const CssmData &data);
	
	/* generate only */
	void final(CssmData &out);	
	
	/* verify only */
	void final(const CssmData &in);	

	size_t outputSize(bool final, size_t inSize);

private:
	CCHmacContext	hmacCtx;
	CSSM_ALGORITHMS	mAlg;
	uint32			mDigestSize;
};

#ifdef	CRYPTKIT_CSP_ENABLE
#include <security_cryptkit/HmacSha1Legacy.h>

/* This version is bug-for-bug compatible with a legacy implementation */

class MacLegacyContext : public AppleCSPContext  {
public:
	MacLegacyContext(
		AppleCSPSession &session,
		CSSM_ALGORITHMS alg) : 
			AppleCSPContext(session), mHmac(NULL) { }
	~MacLegacyContext();
	
	/* called out from CSPFullPluginSession....
	 * both generate and verify: */
	void init(const Context &context, bool isSigning);
	void update(const CssmData &data);
	
	/* generate only */
	void final(CssmData &out);	
	
	/* verify only */
	void final(const CssmData &in);	

	size_t outputSize(bool final, size_t inSize);

private:
	hmacLegacyContextRef	mHmac;
};

#endif	/* CRYPTKIT_CSP_ENABLE */

#endif	/* _MAC_CONTEXT_H_ */
