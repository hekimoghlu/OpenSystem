/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 20, 2022.
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
HEIMCRED_CONST(CFTypeRef, kHEIMAttrType); /* kHEIMAttrType */

HEIMCRED_CONST(CFStringRef, kHEIMTypeGeneric);
HEIMCRED_CONST(CFStringRef, kHEIMTypeKerberos);
HEIMCRED_CONST(CFStringRef, kHEIMTypeIAKerb);
HEIMCRED_CONST(CFStringRef, kHEIMTypeNTLM);
HEIMCRED_CONST(CFStringRef, kHEIMTypeConfiguration);
HEIMCRED_CONST(CFStringRef, kHEIMTypeSchema);
HEIMCRED_CONST(CFStringRef, kHEIMTypeKerberosAcquireCred);
HEIMCRED_CONST(CFStringRef, kHEIMTypeNTLMRelfection);
HEIMCRED_CONST(CFStringRef, kHEIMTypeSCRAM);

/* schema types */
HEIMCRED_CONST(CFStringRef, kHEIMObjectType);
HEIMCRED_CONST(CFStringRef, kHEIMObjectKerberos);
HEIMCRED_CONST(CFStringRef, kHEIMObjectNTLM);
HEIMCRED_CONST(CFStringRef, kHEIMObjectGeneric);
HEIMCRED_CONST(CFStringRef, kHEIMObjectConfiguration);
HEIMCRED_CONST(CFStringRef, kHEIMObjectKerberosAcquireCred);
HEIMCRED_CONST(CFStringRef, kHEIMObjectNTLMReflection);
HEIMCRED_CONST(CFStringRef, kHEIMObjectSCRAM);
HEIMCRED_CONST(CFStringRef, kHEIMObjectAny);

HEIMCRED_CONST(CFTypeRef, kHEIMAttrClientName);
HEIMCRED_CONST(CFStringRef, kHEIMNameUserName);
HEIMCRED_CONST(CFStringRef, kHEIMNameMechKerberos);
HEIMCRED_CONST(CFStringRef, kHEIMNameMechIAKerb);
HEIMCRED_CONST(CFStringRef, kHEIMNameMechNTLM);

HEIMCRED_CONST(CFTypeRef, kHEIMAttrServerName); /* CFDict of types generic + mech names */

HEIMCRED_CONST(CFTypeRef, kHEIMAttrUUID);

HEIMCRED_CONST(CFTypeRef, kHEIMAttrDisplayName);

HEIMCRED_CONST(CFTypeRef, kHEIMAttrCredential);	/* CFBooleanRef */
HEIMCRED_CONST(CFStringRef, kHEIMCredentialPassword);
HEIMCRED_CONST(CFStringRef, kHEIMCredentialCertificate);

HEIMCRED_CONST(CFTypeRef, kHEIMAttrLeadCredential); /* CFBooleanRef */
HEIMCRED_CONST(CFTypeRef, kHEIMAttrParentCredential); /* CFUUIDRef */

HEIMCRED_CONST(CFTypeRef, kHEIMAttrData); /* CFDataRef */

HEIMCRED_CONST(CFTypeRef, kHEIMAttrTransient);

HEIMCRED_CONST(CFTypeRef, kHEIMAttrAllowedDomain); /* CFArray[match rules] */

HEIMCRED_CONST(CFTypeRef, kHEIMAttrStatus);
HEIMCRED_CONST(CFStringRef, kHEIMStatusInvalid);
HEIMCRED_CONST(CFStringRef, kHEIMStatusCanRefresh);
HEIMCRED_CONST(CFStringRef, kHEIMStatusValid);

HEIMCRED_CONST(CFTypeRef, kHEIMAttrStoreTime); /* CFDateRef */
HEIMCRED_CONST(CFTypeRef, kHEIMAttrAuthTime); /* CFDateRef */
HEIMCRED_CONST(CFTypeRef, kHEIMAttrExpire); /* CFDateRef */
HEIMCRED_CONST(CFTypeRef, kHEIMAttrRenewTill); /* CFDateRef */

HEIMCRED_CONST(CFTypeRef, kHEIMAttrRetainStatus); /* CFNumberRef */

HEIMCRED_CONST(CFTypeRef, kHEIMAttrBundleIdentifierACL); /* CFArray[bundle-id] */

HEIMCRED_CONST(CFTypeRef, kHEIMAttrDefaultCredential); /* BooleanRef */
HEIMCRED_CONST(CFTypeRef, kHEIMAttrTemporaryCache); /* BooleanRef */
#ifdef ENABLE_KCM_COMPAT
HEIMCRED_CONST(CFTypeRef, kHEIMAttrCompatabilityCache); /* BooleanRef */
#endif

HEIMCRED_CONST(CFTypeRef, kHEIMAttrKerberosTicketGrantingTicket); /* BooleanRef */

HEIMCRED_CONST(CFTypeRef, kHEIMAttrAltDSID); /* Unique User Id for Shared iPad */

HEIMCRED_CONST(CFTypeRef, kHEIMAttrUserID); /*  User Id for use-uid-matching */
HEIMCRED_CONST(CFTypeRef, kHEIMAttrASID); /*  asid for use-uid-matching */

/* NTLM */
HEIMCRED_CONST(CFStringRef, kHEIMAttrNTLMUsername);
HEIMCRED_CONST(CFStringRef, kHEIMAttrNTLMDomain);
HEIMCRED_CONST(CFStringRef, kHEIMAttrNTLMChannelBinding);
HEIMCRED_CONST(CFStringRef, kHEIMAttrNTLMChallenge);
HEIMCRED_CONST(CFStringRef, kHEIMAttrNTLMType1Data);
HEIMCRED_CONST(CFStringRef, kHEIMAttrNTLMType2Data);
HEIMCRED_CONST(CFStringRef, kHEIMAttrNTLMType3Data);
HEIMCRED_CONST(CFStringRef, kHEIMAttrNTLMClientTargetName);
HEIMCRED_CONST(CFStringRef, kHEIMAttrNTLMClientFlags);
HEIMCRED_CONST(CFStringRef, kHEIMAttrNTLMSessionKey);
HEIMCRED_CONST(CFStringRef, kHEIMAttrNTLMKCMFlags);
HEIMCRED_CONST(CFTypeRef, kHEIMAttrLabelValue); /* CFDataRef */

/* SCRAM */
HEIMCRED_CONST(CFStringRef, kHEIMAttrSCRAMUsername);
HEIMCRED_CONST(CFStringRef, kHEIMAttrSCRAMIterations);
HEIMCRED_CONST(CFStringRef, kHEIMAttrSCRAMSalt);
HEIMCRED_CONST(CFStringRef, kHEIMAttrSCRAMC1);
HEIMCRED_CONST(CFStringRef, kHEIMAttrSCRAMS1);
HEIMCRED_CONST(CFStringRef, kHEIMAttrSCRAMC2NoProof);
HEIMCRED_CONST(CFStringRef, kHEIMAttrSCRAMProof);
HEIMCRED_CONST(CFStringRef, kHEIMAttrSCRAMServer);
HEIMCRED_CONST(CFStringRef, kHEIMAttrSCRAMSessionKey);

#define HEIMCRED_NTLM_FLAG_AV_GUEST 8
