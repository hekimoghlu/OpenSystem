/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 8, 2021.
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
// cryptkitcsp - top C++ implementation layer for CryptKit
//

#ifdef	CRYPTKIT_CSP_ENABLE

#include "cryptkitcsp.h"
#include "FEESignatureObject.h"			/* raw signer */
#include <SignatureContext.h>
#include "FEEKeys.h"
#include "FEEAsymmetricContext.h"
#include <Security/cssmapple.h>
#include <security_cryptkit/falloc.h>
#include <security_cryptkit/feeFunctions.h>
#include <SHA1_MD5_Object.h>
#include <SHA2_Object.h>
#include <security_cdsa_utilities/digestobject.h>

Allocator *CryptKitFactory::normAllocator;
Allocator *CryptKitFactory::privAllocator;

/*
 * CryptKit-style memory allocator callbacks
 */
static void *ckMalloc(unsigned size)
{
	return CryptKitFactory::privAllocator->malloc(size);
}
static void ckFree(void *data)
{
	CryptKitFactory::privAllocator->free(data);
}
static void *ckRealloc(void *oldPtr, unsigned newSize)
{
	return CryptKitFactory::privAllocator->realloc(oldPtr, newSize);
}

//
// Manage the CryptKit algorithm factory
//

CryptKitFactory::CryptKitFactory(Allocator *normAlloc, Allocator *privAlloc)
{
	setNormAllocator(normAlloc);
	setPrivAllocator(privAlloc);
	/* once-per-address space */
	initCryptKit();
	fallocRegister(ckMalloc, ckFree, ckRealloc);
}

CryptKitFactory::~CryptKitFactory()
{
	terminateCryptKit();
}

bool CryptKitFactory::setup(
	AppleCSPSession &session,	
	CSPFullPluginSession::CSPContext * &cspCtx, 
	const Context &context)
{
	switch(context.type()) {
		case CSSM_ALGCLASS_SIGNATURE:
			switch(context.algorithm()) {
				case CSSM_ALGID_FEE_MD5:
					if(cspCtx == NULL) {
						cspCtx = new SignatureContext(session,
							*(new MD5Object()),
							*(new FEERawSigner(feeRandCallback, 
								&session,
								session,
								*privAllocator)));
					}
					return true;
				case CSSM_ALGID_FEE_SHA1:
					if(cspCtx == NULL) {
						cspCtx = new SignatureContext(session,
							*(new SHA1Object()),
							*(new FEERawSigner(feeRandCallback, 
								&session,
								session,
								*privAllocator)));
					}
					return true;
				case CSSM_ALGID_SHA1WithECDSA:
					if(cspCtx == NULL) {
						cspCtx = new SignatureContext(session,
							*(new SHA1Object()),
							*(new FEEECDSASigner(feeRandCallback, 
								&session,
								session,
								*privAllocator)));
					}
					return true;
				case CSSM_ALGID_SHA224WithECDSA:
					if(cspCtx == NULL) {
						cspCtx = new SignatureContext(session,
							*(new SHA224Object()),
							*(new FEEECDSASigner(feeRandCallback, 
								&session,
								session,
								*privAllocator)));
					}
					return true;
				case CSSM_ALGID_SHA256WithECDSA:
					if(cspCtx == NULL) {
						cspCtx = new SignatureContext(session,
							*(new SHA256Object()),
							*(new FEEECDSASigner(feeRandCallback, 
								&session,
								session,
								*privAllocator)));
					}
					return true;
				case CSSM_ALGID_SHA384WithECDSA:
					if(cspCtx == NULL) {
						cspCtx = new SignatureContext(session,
							*(new SHA384Object()),
							*(new FEEECDSASigner(feeRandCallback, 
								&session,
								session,
								*privAllocator)));
					}
					return true;
				case CSSM_ALGID_SHA512WithECDSA:
					if(cspCtx == NULL) {
						cspCtx = new SignatureContext(session,
							*(new SHA512Object()),
							*(new FEEECDSASigner(feeRandCallback, 
								&session,
								session,
								*privAllocator)));
					}
					return true;

				case CSSM_ALGID_FEE:
					if(cspCtx == NULL) {
						cspCtx = new SignatureContext(session,
							*(new NullDigest()),
							*(new FEERawSigner(feeRandCallback, 
								&session,
								session,
								*privAllocator)));
					}
					return true;
				case CSSM_ALGID_ECDSA:
					if(cspCtx == NULL) {
						cspCtx = new SignatureContext(session,
							*(new NullDigest()),
							*(new FEEECDSASigner(feeRandCallback, 
								&session,
								session,
								*privAllocator)));
					}
					return true;
				default:
					break;
			}
			break;		

		case CSSM_ALGCLASS_KEYGEN:
			switch(context.algorithm()) {
				case CSSM_ALGID_FEE:
				case CSSM_ALGID_ECDSA:
					if(cspCtx == NULL) {
						cspCtx = new CryptKit::FEEKeyPairGenContext(session, context);
					}
					return true;
				default:
					break;
			}
			break;		

		case CSSM_ALGCLASS_ASYMMETRIC:
			switch(context.algorithm()) {
				case CSSM_ALGID_FEEDEXP:
					if(cspCtx == NULL) {
						cspCtx = new CryptKit::FEEDExpContext(session);
					}
					return true;
				case CSSM_ALGID_FEED:
					if(cspCtx == NULL) {
						cspCtx = new CryptKit::FEEDContext(session);
					}
					return true;
				default:
					break;
			}
			break;		
		
		/* more here - symmetric, etc. */
		default:
			break;
	}
	/* not implemented here */
	return false;
}

#endif	/* CRYPTKIT_CSP_ENABLE */


