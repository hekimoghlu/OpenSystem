/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 19, 2023.
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
/*!
    @header SecCmsSignedData.h

    @availability 10.4 and later
    @abstract Interfaces of the CMS implementation.
    @discussion The functions here implement functions for encoding
                and decoding Cryptographic Message Syntax (CMS) objects
                as described in rfc3369.
 */

#ifndef _SECURITY_SECCMSSIGNEDDATA_H_
#define _SECURITY_SECCMSSIGNEDDATA_H_  1

#include <Security/SecCmsBase.h>
#include <Security/SecTrust.h>

__BEGIN_DECLS

/*!
    @function
    @abstract Create a new SecCmsSignedData object.
    @param cmsg Pointer to a SecCmsMessage in which this SecCmsSignedData
        should be created.
 */
extern SecCmsSignedDataRef
SecCmsSignedDataCreate(SecCmsMessageRef cmsg);

/*!
    @function
 */
extern void
SecCmsSignedDataDestroy(SecCmsSignedDataRef sigd);

/*!
    @function
    @abstract Retrieve the SignedData's signer list.
 */
extern SecCmsSignerInfoRef *
SecCmsSignedDataGetSignerInfos(SecCmsSignedDataRef sigd);

/*!
    @function
 */
extern int
SecCmsSignedDataSignerInfoCount(SecCmsSignedDataRef sigd);

/*!
    @function
 */
extern SecCmsSignerInfoRef
SecCmsSignedDataGetSignerInfo(SecCmsSignedDataRef sigd, int i);

/*!
    @function
    @abstract Retrieve the SignedData's digest algorithm list.
 */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
extern SECAlgorithmID **
SecCmsSignedDataGetDigestAlgs(SecCmsSignedDataRef sigd);
#pragma clang diagnostic pop

/*!
    @function
    @abstract Return pointer to this signedData's contentinfo.
 */
extern SecCmsContentInfoRef
SecCmsSignedDataGetContentInfo(SecCmsSignedDataRef sigd);

/*!
    @function
    @discussion XXX Should be obsoleted.
 */
extern OSStatus
SecCmsSignedDataImportCerts(SecCmsSignedDataRef sigd, SecKeychainRef keychain,
				SECCertUsage certusage, Boolean keepcerts)
API_DEPRECATED_WITH_REPLACEMENT("SecItemAdd", macos(10.0, 12.0), ios(1.0, 15.0), watchos(1.0, 8.0), tvos(9.0, 15.0)) API_UNAVAILABLE(macCatalyst);


/*!
    @function
    @abstract See if we have digests in place.
 */
extern Boolean
SecCmsSignedDataHasDigests(SecCmsSignedDataRef sigd);

/*!
    @function
    @abstract Check the signatures.
    @discussion The digests were either calculated during decoding (and are stored in the
                signedData itself) or set after decoding using SecCmsSignedDataSetDigests.

                The verification checks if the signing cert is valid and has a trusted chain
                for the purpose specified by "policies".

                If trustRef is NULL the cert chain is verified and the VerificationStatus is set accordingly.
                Otherwise a SecTrust object is returned for the caller to evaluate using SecTrustEvaluate().
 */
extern OSStatus
SecCmsSignedDataVerifySignerInfo(SecCmsSignedDataRef sigd, int i, SecKeychainRef keychainOrArray,
				 CFTypeRef policies, SecTrustRef *trustRef)
API_DEPRECATED_WITH_REPLACEMENT("SecCmsSignedDataVerifySigner", macos(10.0, 12.0), ios(1.0, 15.0), watchos(1.0, 8.0), tvos(9.0, 15.0)) API_UNAVAILABLE(macCatalyst);

/*!
    @function
    @abstract Check the signatures.
    @discussion The digests were either calculated during decoding (and are stored in the
                signedData itself) or set after decoding using SecCmsSignedDataSetDigests.

                The verification checks if the signing cert is valid and has a trusted chain
                for the purpose specified by "policies".

                If trustRef is NULL the cert chain is verified and the VerificationStatus is set accordingly.
                Otherwise a SecTrust object is returned for the caller to evaluate using SecTrustEvaluate().
 */
extern OSStatus
SecCmsSignedDataVerifySigner(SecCmsSignedDataRef sigd, int i, CFTypeRef policies, SecTrustRef *trustRef);

/*!
    @function
    @abstract Verify the certs in a certs-only message.
*/
extern OSStatus
SecCmsSignedDataVerifyCertsOnly(SecCmsSignedDataRef sigd, 
                                  SecKeychainRef keychainOrArray, 
                                  CFTypeRef policies)
API_DEPRECATED_WITH_REPLACEMENT("SecCmsSignedDataVerifyCertsOnlyMessage", macos(10.0, 12.0), ios(1.0, 15.0), watchos(1.0, 8.0), tvos(9.0, 15.0)) API_UNAVAILABLE(macCatalyst);

/*!
 @function
 @abstract Verify the certs in a certs-only message.
 */
extern OSStatus
SecCmsSignedDataVerifyCertsOnlyMessage(SecCmsSignedDataRef sigd,
                                       CFTypeRef policies);

/*!
    @function
 */
extern OSStatus
SecCmsSignedDataAddCertList(SecCmsSignedDataRef sigd, CFArrayRef certlist);

/*!
    @function
    @abstract Add cert and its entire chain to the set of certs.
 */
extern OSStatus
SecCmsSignedDataAddCertChain(SecCmsSignedDataRef sigd, SecCertificateRef cert);

/*!
    @function
 */
extern OSStatus
SecCmsSignedDataAddCertificate(SecCmsSignedDataRef sigd, SecCertificateRef cert);

/*!
    @function
 */
extern Boolean
SecCmsSignedDataContainsCertsOrCrls(SecCmsSignedDataRef sigd);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#if TARGET_OS_OSX
/*!
     @function
     @abstract Retrieve the SignedData's certificate list.
 */
extern CSSM_DATA_PTR *
SecCmsSignedDataGetCertificateList(SecCmsSignedDataRef sigd)
    API_AVAILABLE(macos(10.4)) API_UNAVAILABLE(macCatalyst);
#else // !TARGET_OS_OSX
/*!
    @function
    @abstract Retrieve the SignedData's certificate list.
 */
extern SecAsn1Item * *
SecCmsSignedDataGetCertificateList(SecCmsSignedDataRef sigd)
    API_AVAILABLE(ios(2.0), tvos(2.0), watchos(1.0)) API_UNAVAILABLE(macCatalyst);
#endif // !TARGET_OS_OSX

#pragma clang diagnostic pop

/*!
    @function
    @abstract Create a certs-only SignedData.
    @param cert Base certificate that will be included
    @param include_chain If true, include the complete cert chain for cert.
    @discussion More certs and chains can be added via AddCertificate and AddCertChain.
    @result An error results in a return value of NULL and an error set.
 */
extern SecCmsSignedDataRef
SecCmsSignedDataCreateCertsOnly(SecCmsMessageRef cmsg, SecCertificateRef cert, Boolean include_chain);

#if TARGET_OS_IPHONE
/*!
	@function
    @abstract Finalize the digests in digestContext and apply them to sigd.
    @param sigd A SecCmsSignedDataRef for which the digests have been calculated
    @param digestContext A digestContext created with SecCmsDigestContextStartMultiple.
	@result The digest will have been applied to sigd.  After this call completes sigd is ready to accept
	SecCmsSignedDataVerifySignerInfo() calls.  The caller should still destroy digestContext with a SecCmsDigestContextDestroy() call.

 */
extern OSStatus SecCmsSignedDataSetDigestContext(SecCmsSignedDataRef sigd,
												 SecCmsDigestContextRef digestContext)
     API_AVAILABLE(ios(2.0), tvos(2.0), watchos(1.0)) API_UNAVAILABLE(macos, macCatalyst);
#endif

#if TARGET_OS_OSX
extern OSStatus
SecCmsSignedDataAddSignerInfo(SecCmsSignedDataRef sigd,
                              SecCmsSignerInfoRef signerinfo);

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
extern OSStatus
SecCmsSignedDataSetDigests(SecCmsSignedDataRef sigd,
                           SECAlgorithmID **digestalgs,
                           CSSM_DATA_PTR *digests)
    API_AVAILABLE(macos(10.4)) API_UNAVAILABLE(macCatalyst);
#pragma clang diagnostic pop
#endif

__END_DECLS

#endif /* _SECURITY_SECCMSSIGNEDDATA_H_ */
