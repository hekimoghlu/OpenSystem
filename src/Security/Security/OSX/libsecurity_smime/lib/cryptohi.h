/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 11, 2025.
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
#ifndef _CRYPTOHI_H_
#define _CRYPTOHI_H_

#include <Security/SecCmsBase.h>
#include <security_asn1/seccomon.h>


SEC_BEGIN_PROTOS


/****************************************/
/*
** DER encode/decode DSA signatures
*/

/* ANSI X9.57 defines DSA signatures as DER encoded data.  Our DSA code (and
 * most of the rest of the world) just generates 40 bytes of raw data.  These
 * functions convert between formats.
 */
//extern SECStatus DSAU_EncodeDerSig(SECItem *dest, SECItem *src);
//extern SECItem *DSAU_DecodeDerSig(SECItem *item);

/*
 * Return a csp handle able to deal with algorithm
 */
extern CSSM_CSP_HANDLE SecCspHandleForAlgorithm(CSSM_ALGORITHMS algorithm);

/*
 * Return a CSSM_ALGORITHMS for a given SECOidTag or 0 if there is none
 */
extern CSSM_ALGORITHMS SECOID_FindyCssmAlgorithmByTag(SECOidTag algTag);


/****************************************/
/*
** Signature creation operations
*/

/*
** Sign a single block of data using private key encryption and given
** signature/hash algorithm.
**	"result" the final signature data (memory is allocated)
**	"buf" the input data to sign
**	"len" the amount of data to sign
**	"pk" the private key to encrypt with
**	"algid" the signature/hash algorithm to sign with 
**		(must be compatible with the key type).
*/
extern SECStatus
SEC_SignData(SECItem* result, unsigned char* buf, int len, SecPrivateKeyRef pk, SECOidTag digAlgTag, SECOidTag sigAlgTag);

/*
** Sign a pre-digested block of data using private key encryption, encoding
**  The given signature/hash algorithm.
**	"result" the final signature data (memory is allocated)
**	"digest" the digest to sign
**	"pk" the private key to encrypt with
**	"algtag" The algorithm tag to encode (need for RSA only)
*/
extern SECStatus
SGN_Digest(SecPrivateKeyRef privKey, SECOidTag digAlgTag, SECOidTag sigAlgTag, SECItem* result, SECItem* digest);

/****************************************/
/*
** Signature verification operations
*/


/*
** Verify the signature on a block of data for which we already have
** the digest. The signature data is an RSA private key encrypted
** block of data formatted according to PKCS#1.
** 	"dig" the digest
** 	"key" the public key to check the signature with
** 	"sig" the encrypted signature data
**	"algid" specifies the signing algorithm to use.  This must match
**	    the key type.
**/
extern SECStatus VFY_VerifyDigest(SECItem* dig,
                                  SecPublicKeyRef key,
                                  SECItem* sig,
                                  SECOidTag digAlgTag,
                                  SECOidTag sigAlgTag,
                                  void* wincx);

/*
** Verify the signature on a block of data. The signature data is an RSA
** private key encrypted block of data formatted according to PKCS#1.
** 	"buf" the input data
** 	"len" the length of the input data
** 	"key" the public key to check the signature with
** 	"sig" the encrypted signature data
**	"algid" specifies the signing algorithm to use.  This must match
**	    the key type.
*/
extern SECStatus VFY_VerifyData(unsigned char* buf,
                                int len,
                                SecPublicKeyRef key,
                                SECItem* sig,
                                SECOidTag digAlgTag,
                                SECOidTag sigAlgTag,
                                void* wincx);


extern SECStatus WRAP_PubWrapSymKey(SecPublicKeyRef publickey, SecSymmetricKeyRef bulkkey, CSSM_DATA_PTR encKey);


extern SecSymmetricKeyRef
WRAP_PubUnwrapSymKey(SecPrivateKeyRef privkey, CSSM_DATA_PTR encKey, SECOidTag bulkalgtag);

CFStringRef SECOID_CopyKeyTypeByTag(SECOidTag tag);

/*
 * NULL wrap a ref key to raw key in default format.
 * The utility of this function is that it rescues the actual data content
 * of the SecSymmetricKeyRef so that we can encrypt it.
 */
OSStatus cmsNullWrapKey(SecKeyRef refKey, CSSM_KEY_PTR rawKey);

SEC_END_PROTOS

#endif /* _CRYPTOHI_H_ */
