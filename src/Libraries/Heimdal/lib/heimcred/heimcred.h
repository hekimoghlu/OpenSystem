/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 9, 2025.
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
#ifndef HEIMDAL_HEIMCRED_H
#define HEIMDAL_HEIMCRED_H 1

#include <CoreFoundation/CoreFoundation.h>
#include <dispatch/dispatch.h>

#define HEIMCRED_CONST(_t,_c) extern const _t _c
#include <heimcred-const.h>
#undef HEIMCRED_CONST

#define CFRELEASE_NULL(x) do { if (x!=NULL) { CFRelease(x); x = NULL; } } while(0)

#if HEIMCRED_SERVER
struct peer;
#endif
/*
 *
 */

typedef struct HeimCred_s *HeimCredRef;

HeimCredRef
HeimCredCreate(CFDictionaryRef attributes, CFErrorRef *error);

CFUUIDRef
HeimCredGetUUID(HeimCredRef);

CFDictionaryRef
HeimCredGetAttributes(HeimCredRef);

HeimCredRef
HeimCredCopyFromUUID(CFUUIDRef);

bool
HeimCredSetAttribute(HeimCredRef cred, CFTypeRef key, CFTypeRef value, CFErrorRef *error);

bool
HeimCredSetAttributes(HeimCredRef cred, CFDictionaryRef attributes, CFErrorRef *error);

CFDictionaryRef
HeimCredCopyAttributes(HeimCredRef cred, CFSetRef attributes, CFErrorRef *error);

CFTypeRef
HeimCredCopyAttribute(HeimCredRef cred, CFTypeRef attribute);

CFArrayRef
HeimCredCopyQuery(CFDictionaryRef query);

bool
HeimCredDeleteQuery(CFDictionaryRef query, CFErrorRef *error);

void
HeimCredDelete(HeimCredRef item);

void
HeimCredDeleteByUUID(CFUUIDRef uuid);

void
HeimCredRetainTransient(HeimCredRef cred);

void
HeimCredReleaseTransient(HeimCredRef cred);

bool
HeimCredMove(CFUUIDRef from, CFUUIDRef to);

CFUUIDRef
HeimCredCopyDefaultCredential(CFStringRef mech, CFErrorRef *error);

CFDictionaryRef
HeimCredCopyStatus(CFStringRef mech);

CFDictionaryRef
HeimCredDoAuth(HeimCredRef cred, CFDictionaryRef attributes, CFErrorRef *error);

bool
HeimCredDeleteAll(CFStringRef altDSID, CFErrorRef *error);

bool
HeimCredAddNTLMChallenge(uint8_t chal[8]);

bool
HeimCredCheckNTLMChallenge(uint8_t challenge[8]);

CFDictionaryRef
HeimCredDoSCRAM(HeimCredRef cred, CFDictionaryRef attributes, CFErrorRef *error);

/*
 * Only valid client side
 */

void
HeimCredSetImpersonateBundle(CFStringRef bundle);

const char *
HeimCredGetImpersonateBundle(void);

void
HeimCredSetImpersonateAuditToken(CFDataRef auditToken) API_AVAILABLE(macos(10.16));

CFDataRef
HeimCredGetImpersonateAuditToken(void) API_AVAILABLE(macos(10.16));


// Use for automated tests only, not for normal use.
void
_HeimCredResetLocalCache(void);

#if HEIMCRED_SERVER
/*
 * Only valid server side side
 */
typedef CFTypeRef (*HeimCredStatusCallback)(HeimCredRef);
typedef void (*HeimCredNotifyCaches)(void);
typedef CFDictionaryRef(*HeimCredTraceCallback)(CFDictionaryRef);

void
_HeimCredRegisterKerberos(void);

void
_HeimCredRegisterNTLM(void);

void
_HeimCredRegisterNTLMReflection(void);

void
_HeimCredRegisterSCRAM(void);

void
_HeimCredRegisterKerberosAcquireCred(void);

CFMutableDictionaryRef
_HeimCredCreateBaseSchema(CFStringRef objectType);

#endif /* HEIMCRED_SERVER */
/*
typedef struct HeimAuth_s *HeimAuthRef;

HeimAuthRef
HeimCreateAuthetication(CFDictionaryRef input);

bool
HeimAuthStep(HeimAuthRef cred, CFTypeRef input, CFTypeRef *output, CFErrorRef *error);
*/

#endif /* HEIMDAL_HEIMCRED_H */
