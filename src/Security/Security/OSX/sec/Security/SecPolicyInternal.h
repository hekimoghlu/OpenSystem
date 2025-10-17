/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 3, 2022.
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
    @header SecPolicyPriv
    The functions provided in SecPolicyInternal provide the interface to
    trust policies used by SecTrust.
*/

#ifndef _SECURITY_SECPOLICYINTERNAL_H_
#define _SECURITY_SECPOLICYINTERNAL_H_

#include <xpc/xpc.h>

#include <Security/SecPolicy.h>
#include <Security/SecTrust.h>
#include <CoreFoundation/CFArray.h>
#include <CoreFoundation/CFString.h>
#include <CoreFoundation/CFRuntime.h>

__BEGIN_DECLS

/********************************************************
 ****************** SecPolicy struct ********************
 ********************************************************/
struct __SecPolicy {
    CFRuntimeBase		_base;
    CFStringRef			_oid;
    CFStringRef		_name;
	CFDictionaryRef		_options;
};

CF_RETURNS_RETAINED SecPolicyRef SecPolicyCreate(CFStringRef oid, CFStringRef name, CFDictionaryRef options);

CFDictionaryRef SecPolicyGetOptions(SecPolicyRef policy);

XPC_RETURNS_RETAINED xpc_object_t SecPolicyArrayCopyXPCArray(CFArrayRef policies, CFErrorRef *error);

CF_RETURNS_RETAINED CFArrayRef SecPolicyArrayCreateDeserialized(CFArrayRef serializedPolicies);
CF_RETURNS_RETAINED CFArrayRef SecPolicyArrayCreateSerialized(CFArrayRef policies);

void SecPolicySetOptionsValue_internal(SecPolicyRef policy, CFStringRef key, CFTypeRef value);

/*
 * MARK: SecLeafPVC functions
 */

typedef struct OpaqueSecLeafPVC *SecLeafPVCRef;

struct OpaqueSecLeafPVC {
    SecCertificateRef leaf;
    CFArrayRef policies;
    CFAbsoluteTime verifyTime;
    CFArrayRef details;
    CFMutableDictionaryRef info;
    CFDictionaryRef callbacks;
    CFIndex policyIX;
    bool result;
};

void SecLeafPVCInit(SecLeafPVCRef pvc, SecCertificateRef leaf, CFArrayRef policies, CFAbsoluteTime verifyTime);
void SecLeafPVCDelete(SecLeafPVCRef pvc);
bool SecLeafPVCLeafChecks(SecLeafPVCRef pvc);

__END_DECLS

#endif /* !_SECURITY_SECPOLICYINTERNAL_H_ */
