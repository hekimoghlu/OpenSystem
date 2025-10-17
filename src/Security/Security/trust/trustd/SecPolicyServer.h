/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, July 28, 2023.
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
    @header SecPolicyServer
    The functions provided in SecPolicyServer.h provide an interface to
    trust policies dealing with certificate revocation.
*/

#ifndef _SECURITY_SECPOLICYSERVER_H_
#define _SECURITY_SECPOLICYSERVER_H_

#include <Security/SecTrust.h>
#include "Security/SecPolicyInternal.h"
#include <Security/SecTrustSettings.h>

#include "trust/trustd/SecTrustServer.h"
#include "trust/trustd/SecCertificateServer.h"

__BEGIN_DECLS

#define kSecPolicySHA256Size 32

void SecPVCInit(SecPVCRef pvc, SecPathBuilderRef builder, CFArrayRef policies);
void SecPVCDelete(SecPVCRef pvc);
void SecPVCSetPath(SecPVCRef pvc, SecCertificatePathVCRef path);
SecPolicyRef SecPVCGetPolicy(SecPVCRef pv);
SecCertificateRef SecPVCGetCertificateAtIndex(SecPVCRef pvc, CFIndex ix);
CFIndex SecPVCGetCertificateCount(SecPVCRef pvc);
CFAbsoluteTime SecPVCGetVerifyTime(SecPVCRef pvc);

/* Set the string result as the reason for the sub policy check key
   failing.  The policy check function should continue processing if
   this function returns true. */
bool SecPVCSetResult(SecPVCRef pv, CFStringRef key, CFIndex ix, CFTypeRef result);
bool SecPVCSetResultForced(SecPVCRef pvc, CFStringRef key, CFIndex ix, CFTypeRef result, bool force);
bool SecPVCSetResultForcedWithTrustResult(SecPVCRef pvc, CFStringRef key, CFIndex ix, CFTypeRef result, bool force,
                                          SecTrustResultType overrideDefaultTR);

/* Is the current result considered successful. */
bool SecPVCIsOkResult(SecPVCRef pvc);

/* Compute details */
void SecPVCComputeDetails(SecPVCRef pvc, SecCertificatePathVCRef path);

/* Run static leaf checks on the path in pvc. */
SecTrustResultType SecPVCLeafChecks(SecPVCRef pvc);

/* Run static parent checks on the path in pvc. */
bool SecPVCParentCertificateChecks(SecPVCRef pvc, CFIndex ix);

/* Run dynamic checks on the complete path in pvc.  Return true if the
   operation is complete, returns false if an async backgroup request was
   scheduled.  Upon completion of the async background job
   SecPathBuilderStep() should be called. */
void SecPVCPathChecks(SecPVCRef pvc);

/* Check whether revocation responses were received for certificates
 * in the path in pvc. If a valid response was not obtained for a
 * certificate, this sets the appropriate error result if revocation
 * was required, and/or definitive revocation info is present. */
void SecPVCPathCheckRevocationResponsesReceived(SecPVCRef pvc);

typedef void (*SecPolicyCheckFunction)(SecPVCRef pv, CFStringRef key);

/*
 * Used by SecTrust to verify if a particular certificate chain matches
 * this policy.  Returns true if the policy accepts the certificate chain.
*/
bool SecPolicyValidate(SecPolicyRef policy, SecPVCRef pvc, CFStringRef key);

void SecPolicyServerInitialize(void);

bool SecPolicyIsEVPolicy(const DERItem *policyOID);

bool SecPVCIsAnchorPerConstraints(SecPVCRef pvc, SecCertificateSourceRef source, SecCertificateRef certificate);

__END_DECLS

#endif /* !_SECURITY_SECPOLICYSERVER_H_ */
