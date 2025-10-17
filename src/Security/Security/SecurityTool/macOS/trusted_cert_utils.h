/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 23, 2023.
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
#ifndef _TRUSTED_CERT_UTILS_H_
#define _TRUSTED_CERT_UTILS_H_  1

#include <Security/SecCertificate.h>
#include <Security/SecPolicy.h>
#include <Security/SecTrust.h>
#include <Security/SecTrustedApplication.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CFRELEASE(cf)	if(cf != NULL) { CFRelease(cf); }

extern void indentIncr(void);
extern void indentDecr(void);
extern void indent(void);
void printAscii(const char *buf, unsigned len, unsigned maxLen);
void printHex(const unsigned char *buf, unsigned len, unsigned maxLen);
void printCfStr(CFStringRef cfstr);
void printCFDate(CFDateRef dateRef);
void printCfNumber(CFNumberRef cfNum);
void printResultType(CFNumberRef cfNum);
void printKeyUsage(CFNumberRef cfNum);
void printCssmErr(CFNumberRef cfNum);
void printCertLabel(SecCertificateRef certRef);
void printCertDescription(SecCertificateRef certRef);
void printCertText(SecCertificateRef certRef);
void printCertChain(SecTrustRef trustRef, bool printPem, bool printText);

/* convert an OID to a SecPolicyRef */
extern SecPolicyRef oidToPolicy(const CSSM_OID *oid);

/* convert a policy string to a SecPolicyRef */
extern SecPolicyRef oidStringToPolicy(const char *oidStr);

/* CSSM_OID --> OID string */
extern const char *oidToOidString(const CSSM_OID *oid);

/* compare OIDs; returns 1 if identical, else returns 0 */
extern int compareOids(const CSSM_OID *oid1, const CSSM_OID *oid2);

/* app path string to SecTrustedApplicationRef */
extern SecTrustedApplicationRef appPathToAppRef(const char *appPath);

/* read a file --> SecCertificateRef */
int readCertFile(const char *fileName, SecCertificateRef *certRef);

#if TARGET_OS_OSX
/* policy string --> CSSM_OID */
const CSSM_OID *policyStringToOid(const char *policy, bool *useTLS);
#endif

/* revocation option string --> revocation option flag */
CFOptionFlags revCheckOptionStringToFlags(const char *revCheckOption);

#ifdef __cplusplus
}
#endif

#endif /* _TRUSTED_CERT_UTILS_H_ */
