/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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
 @header SecKeyInternal
 */

#ifndef _SECURITY_SECKEYINTERNAL_H_
#define _SECURITY_SECKEYINTERNAL_H_

#include <Security/SecBase.h>

#include <Security/SecKeyPriv.h>
#include <corecrypto/ccrng.h>

__BEGIN_DECLS

#if TARGET_OS_OSX
void SecKeySetAuxilliaryCDSAKeyForKey(SecKeyRef cf, SecKeyRef auxKey);
SecKeyRef SecKeyCopyAuxilliaryCDSAKeyForKey(SecKeyRef cf);
#endif

struct ccrng_state *ccrng_seckey(void);

enum {
    // Keep in sync with SecKeyOperationType enum in SecKey.h and SecKeyPriv.h
    kSecKeyOperationTypeCount = 7
};

typedef struct {
    SecKeyRef key;
    SecKeyOperationType operation;
    CFMutableArrayRef algorithm;
    SecKeyOperationMode mode;
} SecKeyOperationContext;

typedef CFTypeRef (*SecKeyAlgorithmAdaptor)(SecKeyOperationContext *context, CFTypeRef in1, CFTypeRef in2, CFErrorRef *error);

void SecKeyOperationContextDestroy(SecKeyOperationContext *context);
CFTypeRef SecKeyRunAlgorithmAndCopyResult(SecKeyOperationContext *context, CFTypeRef in1, CFTypeRef in2, CFErrorRef *error);
SecKeyAlgorithmAdaptor SecKeyGetAlgorithmAdaptor(SecKeyOperationType operation, SecKeyAlgorithm algorithm);

void _SecKeyCheck(SecKeyRef key, const char *callerName);
#define SecKeyCheck(key) _SecKeyCheck(key, __func__)

bool _SecKeyErrorPropagate(bool succeeded, const char *logCallerName, CFErrorRef possibleError CF_CONSUMED, CFErrorRef *error);
#define SecKeyErrorPropagate(s, pe, e) _SecKeyErrorPropagate(s, __func__, pe, e)


__END_DECLS

#endif /* !_SECURITY_SECKEYINTERNAL_H_ */
