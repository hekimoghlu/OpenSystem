/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 5, 2024.
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
#import "remotesigner.h"
#import "cfmunge.h"

#import <MessageSecurity/MessageSecurity.h>
#import <Security/CMSEncoder.h>
#import <Security/SecCmsBase.h>


// A local interface to prevent compilation issues for interfaces not in
// the SDK yet.
@interface MSCMSSignerInfo (LocalBuild)
- (nullable instancetype)initWithCertificate:(SecCertificateRef)certificate
						  signatureAlgorithm:(MSOID * _Nullable)signatureAlgorithm
					useIssuerAndSerialNumber:(BOOL)useIssuerAndSerialNumber
									   error:(NSError * _Nullable __autoreleasing * _Nullable)error;
@end

namespace Security {
namespace CodeSigning {

static SecCSDigestAlgorithm
mapDigestAlgorithm(const MSOIDString msDigestAlgorithm)
{
	if ([msDigestAlgorithm isEqualToString:MSDigestAlgorithmSHA1]) {
		return kSecCodeSignatureHashSHA1;
	} else if ([msDigestAlgorithm isEqualToString:MSDigestAlgorithmSHA256]) {
		return kSecCodeSignatureHashSHA256;
	} else if ([msDigestAlgorithm isEqualToString:MSDigestAlgorithmSHA384]) {
		return kSecCodeSignatureHashSHA384;
	} else if ([msDigestAlgorithm isEqualToString:MSDigestAlgorithmSHA512]) {
		return kSecCodeSignatureHashSHA512;
	}
	return kSecCodeSignatureNoHash;
}

static NSDictionary *
createHashAgilityV2Dictionary(NSDictionary *hashes)
{
	// Converts the hash dictionary provided by the signing flow to one that uses
	// the appropriate keys from digest algorithm OID strings...so just iterate
	// the input dictionary and map the old keys to new keys.
	NSMutableDictionary *output = [NSMutableDictionary dictionary];
	for (NSNumber *key in hashes) {
		MSOIDString newKey = nil;
		switch (key.intValue) {
			case SEC_OID_SHA1: 		newKey = MSDigestAlgorithmSHA1; 	break;
			case SEC_OID_SHA256: 	newKey = MSDigestAlgorithmSHA256; 	break;
			case SEC_OID_SHA384: 	newKey = MSDigestAlgorithmSHA384; 	break;
			case SEC_OID_SHA512: 	newKey = MSDigestAlgorithmSHA512; 	break;
			default:
				secerror("Unexpected digest algorithm: %@", key);
				return nil;
		}
		output[newKey] = hashes[key];
	}
	return output;
}

static NSData *
createHashAgilityV1Data(CFArrayRef hashList)
{
	CFTemp<CFDictionaryRef> v1HashDict("{cdhashes=%O}", hashList);
	CFRef<CFDataRef> hashAgilityV1Attribute = makeCFData(v1HashDict.get());
	return (__bridge NSData *)hashAgilityV1Attribute.get();
}

OSStatus
doRemoteSigning(const CodeDirectory *cd,
				CFDictionaryRef hashDict,
				CFArrayRef hashList,
				CFAbsoluteTime signingTime,
				CFArrayRef certificateChain,
				SecCodeRemoteSignHandler signHandler,
				CFDataRef *outputCMS)
{
	NSError *error = nil;
	MSCMSSignedData *signedData = nil;
	MSCMSSignerInfo *signerInfo = nil;
	CFRef<SecCertificateRef> firstCert;

	// Verify all inputs are valid.
	if (cd == NULL || cd->length() == 0) {
		secerror("Remote signing requires valid code directory.");
		return errSecParam;
	} else if (outputCMS == NULL) {
		secerror("Remote signing requires output CMS parameter.");
		return errSecParam;
	} else if (hashDict == NULL) {
		secerror("Remote signing requires hash dictionary.");
		return errSecParam;
	} else if (hashList == NULL) {
		secerror("Remote signing requires hash list.");
		return errSecParam;
	} else if (signHandler == NULL) {
		secerror("Remote signing requires signing block.");
		return errSecParam;
	} else if (certificateChain == NULL || CFArrayGetCount(certificateChain) == 0) {
		secerror("Unable to perform remote signing with no certificates: %@", certificateChain);
		return errSecParam;
	}

	// Make a signer info with the identity above and using a SHA256 digest algorithm.
	firstCert = (SecCertificateRef)CFArrayGetValueAtIndex(certificateChain, 0);
	MSOID *signingAlgoOID = [MSOID OIDWithString:MSSignatureAlgorithmRSAPKCS1v5SHA256 error:&error];
	if (!signingAlgoOID) {
		secerror("Unable to create signing algorithm: %@", error);
		return errSecMemoryError;
	}

	signerInfo = [MSCMSSignerInfo alloc];
	SEL newInit = @selector(initWithCertificate:signatureAlgorithm:useIssuerAndSerialNumber:error:);
	if ([signerInfo respondsToSelector:newInit]) {
		signerInfo = [signerInfo initWithCertificate:firstCert
								  signatureAlgorithm:signingAlgoOID
							useIssuerAndSerialNumber:YES
											   error:&error];
	} else {
		secerror("Unable to create signer due to old CMS interfaces");
		return errSecBadReq;
	}

	if (!signerInfo || error) {
		secerror("Unable to create signer info: %@, %@, %@", firstCert.get(), signingAlgoOID.OIDString, error);
		return errSecCSCMSConstructionFailed;
	}

	// Initialize the top level signed data with detached data.
	NSData *codeDir = [NSData dataWithBytes:cd length:cd->length()];
	signedData = [[MSCMSSignedData alloc] initWithDataContent:codeDir
												   isDetached:YES
													   signer:signerInfo
									   additionalCertificates:(__bridge NSArray *)certificateChain
														error:&error];
	if (!signedData) {
		secerror("Unable to create signed data: %@", error);
		return errSecCSCMSConstructionFailed;
	}

	// Add signing time into attributes, if necessary.
	if (signingTime != 0) {
		NSDate *signingDate = [NSDate dateWithTimeIntervalSinceReferenceDate:signingTime];
		MSCMSSigningTimeAttribute *signingTimeAttribute = [[MSCMSSigningTimeAttribute alloc] initWithSigningTime:signingDate];
		[signerInfo addProtectedAttribute: signingTimeAttribute];
	}

	// Generate hash agility v1 attribute from the hash list.
	NSData *hashAgilityV1Data = createHashAgilityV1Data(hashList);
	MSCMSAppleHashAgilityAttribute *hashAgility = [[MSCMSAppleHashAgilityAttribute alloc] initWithHashAgilityValue:hashAgilityV1Data];
	[signerInfo addProtectedAttribute: hashAgility];

	// Pass in the hash dictionary to generate the hash agility v2 attribute.
	NSDictionary *hashAgilityV2Dict = createHashAgilityV2Dictionary((__bridge NSDictionary *)hashDict);
	MSCMSAppleHashAgilityV2Attribute *hashAgility2 = [[MSCMSAppleHashAgilityV2Attribute alloc] initWithHashAgilityValues:hashAgilityV2Dict];
	[signerInfo addProtectedAttribute:hashAgility2];

	// Top level object is a content info with pkcs7 signed data type, embedding the signed data above.
	MSCMSContentInfo *topLevelInfo = [[MSCMSContentInfo alloc] initWithEmbeddedContent:signedData];

	// Calculate the signer digest info.
	NSData *signatureDigest = [signerInfo calculateSignerInfoDigest:&error];
	if (!signatureDigest) {
		secerror("Unable to create signature digest: %@, %@", signerInfo, error);
		return errSecCSCMSConstructionFailed;
	}

	// Calculate the right digest algorithm type to pass to the caller.
	MSAlgorithmIdentifier *digestAlgoID = nil;
	digestAlgoID = [MSAlgorithmIdentifier digestAlgorithmWithSignatureAlgorithm:signerInfo.signatureAlgorithm
																			  error:&error];
	if (!digestAlgoID) {
		secerror("Unable to create digest algorithm: %@, %@", signerInfo, error);
		return errSecCSCMSConstructionFailed;
	}

	SecCSDigestAlgorithm digestAlgorithm = mapDigestAlgorithm(digestAlgoID.algorithm.OIDString);
	if (digestAlgorithm == kSecCodeSignatureNoHash) {
		secerror("Unable to map digest algorithm: %@", digestAlgoID.algorithm.OIDString);
		return errSecCSUnsupportedDigestAlgorithm;
	}

	// Call remote block with message digest, and transfer the ownership to objc ARC object.
	secinfo("remotesigner", "Passing out external digest: %d, %@", digestAlgorithm, signatureDigest);
	NSData *externalSignature = (__bridge_transfer NSData *)signHandler((__bridge CFDataRef)signatureDigest, digestAlgorithm);
	if (!externalSignature) {
		secerror("External block did not provide a signature, failing.");
		return errSecCSRemoteSignerFailed;
	}
	secinfo("remotesigner", "Got external signature blob: %@", externalSignature);

	// Pass the external signature into the signer info so it can be encoded in the output.
	[signerInfo setSignature:externalSignature];

	// Encode the full CMS blob and pass it out.
	NSData *fullCMSSignature = [topLevelInfo encodeMessageSecurityObject:&error];
	if (!fullCMSSignature || error) {
		secerror("Failed to encode signature: %@", error);
		return errSecCSCMSConstructionFailed;
	}

	// Return the signature, bridging with a retain to meet the API that the caller must release it.
	secinfo("remotesigner", "Encoded CMS signature: %@", fullCMSSignature);
	*outputCMS = (__bridge_retained CFDataRef)fullCMSSignature;
	return errSecSuccess;
}

}
}
