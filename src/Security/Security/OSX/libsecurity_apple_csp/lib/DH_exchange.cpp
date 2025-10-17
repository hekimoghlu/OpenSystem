/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 22, 2022.
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
 * DH_exchange.cp - Diffie-Hellman key exchange
 */
 
#include "DH_exchange.h"
#include <Security/cssmerr.h>
#include "DH_utils.h"
#include "DH_keys.h"
#include <strings.h>
#include <opensslUtils/opensslUtils.h>

void DeriveKey_DH (
	const Context &context,
	const CssmData &Param,			// other's public key. may be empty
	CSSM_DATA *keyData,				// mallocd by caller
									// we fill in keyData->Length bytes
	AppleCSPSession &session)
{
	bool mallocdPrivKey;
	size_t privSize;
	
	/* private DH key from context - required */
	DH *privKey = contextToDhKey(context, session, CSSM_ATTRIBUTE_KEY,
		CSSM_KEYCLASS_PRIVATE_KEY, CSSM_KEYUSE_DERIVE, mallocdPrivKey);
	if(privKey == NULL) {
		CssmError::throwMe(CSSMERR_CSP_MISSING_ATTR_KEY);
	}
	cspDhDebug("DeriveKey_DH, privKey %p", privKey);
	privSize = DH_size(privKey);
	if(privSize < keyData->Length) {
		/* we've been asked for more bits than this key can generate */
		CssmError::throwMe(CSSMERR_CSP_UNSUPPORTED_KEY_SIZE);
	}
	
	/*
	 * Public key ("their" key) can come from two places:
	 * -- in the context as a CSSM_ATTRIBUTE_PUBLIC_KEY. This is how 
	 *    public keys in X509 format must be used in this function
	 * -- in the incoming Param, the raw unformatted (PKCS3) form 
	 */
	bool mallocdPubKey = false;
	BIGNUM *pubKeyBn = NULL;
	bool allocdPubKeyBn = false;
	DH *pubKey = contextToDhKey(context, session, CSSM_ATTRIBUTE_PUBLIC_KEY,
		CSSM_KEYCLASS_PUBLIC_KEY, CSSM_KEYUSE_DERIVE, mallocdPubKey);
	if(pubKey != NULL) {
		if(pubKey->pub_key == NULL) {
			errorLog0("DeriveKey_DH: public key in context with no pub_key\n");
			CssmError::throwMe(CSSMERR_CSP_INVALID_KEY);
		}
		pubKeyBn = pubKey->pub_key;
		cspDhDebug("DeriveKey_DH, pubKey from context %p", pubKey);
	}
	else {
		if((Param.Data == NULL) || (Param.Length == 0)) {
			errorLog0("DeriveKey_DH: no pub_key, no Param\n");
			CssmError::throwMe(CSSMERR_CSP_INVALID_KEY);
		}
		pubKeyBn = BN_bin2bn(Param.Data, (int)Param.Length, NULL);
		if(pubKeyBn == NULL) {
			CssmError::throwMe(CSSMERR_CSP_MEMORY_ERROR);
		}
		allocdPubKeyBn = true;
		cspDhDebug("DeriveKey_DH, no pubKey in context");
	}
	unsigned char *buf = (unsigned char *)session.malloc(privSize);
	int rtn = DH_compute_key(buf, pubKeyBn, privKey);
	if(rtn > 0) {
		/*
		 * FIXME : I have not found a specification describing *which*
		 * bytes of the value we just computed we are supposed to
		 * use as the actual key bytes. We use the M.S. bytes.
		 *
		 * Note that due to modulo arithmetic, we may have gotten fewer
		 * bytes than we asked for. If so, the caller will have
		 * to deal with that if they really need privSize bytes.
		 */
		assert((uint32)rtn <= privSize);
		CSSM_SIZE toMove = keyData->Length;
		if((uint32)rtn < toMove) {
			toMove = (uint32)rtn;
		}
		memmove(keyData->Data, buf, toMove);
		keyData->Length = toMove;
	}
	if(mallocdPrivKey) {
		DH_free(privKey);
	}
	if(mallocdPubKey) {
		DH_free(pubKey);
	}
	if(allocdPubKeyBn) {
		BN_free(pubKeyBn);
	}
	session.free(buf);
	if(rtn <= 0) {
		throwRsaDsa("DH_compute_key");
	}
}

