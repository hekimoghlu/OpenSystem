/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 25, 2025.
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
#import "RTCCertificate.h"

#import "base/RTCLogging.h"

#include "rtc_base/logging.h"
#include "rtc_base/rtc_certificate_generator.h"
#include "rtc_base/ssl_identity.h"

@implementation RTCCertificate

@synthesize private_key = _private_key;
@synthesize certificate = _certificate;

- (id)copyWithZone:(NSZone *)zone {
  id copy = [[[self class] alloc] initWithPrivateKey:[self.private_key copyWithZone:zone]
                                         certificate:[self.certificate copyWithZone:zone]];
  return copy;
}

- (instancetype)initWithPrivateKey:(NSString *)private_key certificate:(NSString *)certificate {
  if (self = [super init]) {
    _private_key = [private_key copy];
    _certificate = [certificate copy];
  }
  return self;
}

+ (nullable RTCCertificate *)generateCertificateWithParams:(NSDictionary *)params {
  rtc::KeyType keyType = rtc::KT_ECDSA;
  NSString *keyTypeString = [params valueForKey:@"name"];
  if (keyTypeString && [keyTypeString isEqualToString:@"RSASSA-PKCS1-v1_5"]) {
    keyType = rtc::KT_RSA;
  }

  NSNumber *expires = [params valueForKey:@"expires"];
  rtc::scoped_refptr<rtc::RTCCertificate> cc_certificate = nullptr;
  if (expires != nil) {
    uint64_t expirationTimestamp = [expires unsignedLongLongValue];
    cc_certificate = rtc::RTCCertificateGenerator::GenerateCertificate(rtc::KeyParams(keyType),
                                                                       expirationTimestamp);
  } else {
    cc_certificate =
        rtc::RTCCertificateGenerator::GenerateCertificate(rtc::KeyParams(keyType), absl::nullopt);
  }
  if (!cc_certificate) {
    RTCLogError(@"Failed to generate certificate.");
    return nullptr;
  }
  // grab PEMs and create an NS RTCCerticicate
  rtc::RTCCertificatePEM pem = cc_certificate->ToPEM();
  std::string pem_private_key = pem.private_key();
  std::string pem_certificate = pem.certificate();
  RTC_LOG(LS_INFO) << "CERT PEM ";
  RTC_LOG(LS_INFO) << pem_certificate;

  RTCCertificate *cert = [[RTCCertificate alloc] initWithPrivateKey:@(pem_private_key.c_str())
                                                        certificate:@(pem_certificate.c_str())];
  return cert;
}

@end
