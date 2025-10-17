/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 7, 2023.
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
   SecJWS.h
*/

#import <Foundation/Foundation.h>
#import <Security/Security.h>

NS_ASSUME_NONNULL_BEGIN

extern NSString *const SecJWSErrorDomain;

typedef NS_ERROR_ENUM(SecJWSErrorDomain, SecJWSError) {
    SecJWSErrorInvalidCompactEncoding,            /// Not compact encoding
    SecJWSErrorCompactEncodingParseError,         /// Did not have all 3-tuple compact encoding segments
    SecJWSErrorCompactEncodingClaimError,         /// The Claim body is invalid or unparseable for this specification
    SecJWSErrorCompactEncodingPayloadParseError,  /// Could not parse the BASE64*URL encoding of the payload
    SecJWSErrorCompactEncodingHeaderParseError,   /// Could not parse the BASE64*URL encoding of the header
    SecJWSErrorHeaderFormatParseError,            /// Could not parse the JSON encoding of the header
    SecJWSErrorHeaderIncorrectKeyError,           /// Header contains an incorrect set of keys
    SecJWSErrorHeaderIncorrectAlgorithmError,     /// Header specifies an incorrect signature algorithm
    SecJWSErrorHeaderInvalidKeyIDError,           /// Header specifies a key ID that does not match one from the public DB
    SecJWSErrorHeaderInvalidValueError,           /// Invalid value for an unspecified header
    SecJWSErrorInvalidPublicKey,                  /// The public key is invalid (unable to parse/decode, wrong type, etc.)
    SecJWSErrorSignatureFormatError,              /// The signature has an incorrect format
    SecJWSErrorSignatureVerificationError,        /// Verification of the signature failed
};

@interface SecJWSDecoder : NSObject
@property (nonatomic, readonly) NSString *keyID;
@property (nonatomic, readonly) NSData *payload;
@property (nonatomic, readonly) NSData *signature;
@property (nonatomic, readonly) NSError *verificationError;   /// One of the SecJWSError errors
- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithJWSCompactEncodedString:(NSString *)compactEncodedString keyID:(NSString * _Nullable)keyID publicKey:(SecKeyRef)publicKeyRef NS_DESIGNATED_INITIALIZER;
- (NSData *) dataWithBase64URLEncodedString:(NSString *)base64URLEncodedString;
@end

@interface SecJWSEncoder : NSObject
@property (assign) SecKeyRef publicKey;
@property (assign) SecKeyRef privateKey;
- (instancetype) init;                          /// will generate default EC 256 key pair
- (instancetype) initWithPublicKey:(SecKeyRef)pubKey privateKey:(SecKeyRef)privKey;
- (NSString *) encodedJWSWithPayload:(NSDictionary * _Nullable)payload kid:(NSString * _Nullable)kid nonce:(NSString *)nonce url:(NSString *)url error:(NSError ** _Nullable)error;
- (NSDictionary * _Nullable) jwkPublicKey;
- (NSString *) base64URLEncodedStringRepresentationWithData:(NSData *)data;
- (NSString *) base64URLEncodedStringRepresentationWithDictionary:(NSDictionary *)dictionary;
- (NSString *) compactJSONStringRepresentationWithDictionary:(NSDictionary *)dictionary;
@end

NS_ASSUME_NONNULL_END
