/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 31, 2022.
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
//
//  SOSRecoveryKeyBag.h
//  sec
//

#ifndef SOSRecoveryKeyBag_h
#define SOSRecoveryKeyBag_h

#include <stdio.h>
#include <CoreFoundation/CoreFoundation.h>

typedef struct __OpaqueSOSRecoveryKeyBag *SOSRecoveryKeyBagRef;

CFTypeRef SOSRecoveryKeyBageGetTypeID(void);
SOSRecoveryKeyBagRef SOSRecoveryKeyBagCreateForAccount(CFAllocatorRef allocator, CFTypeRef account, CFDataRef pubData, CFErrorRef *error);
CFDataRef SOSRecoveryKeyCopyKeyForAccount(CFAllocatorRef allocator, CFTypeRef account, SOSRecoveryKeyBagRef recoveryKeyBag, CFErrorRef *error);

CFDataRef SOSRecoveryKeyBagCopyEncoded(SOSRecoveryKeyBagRef RecoveryKeyBag, CFErrorRef* error);
SOSRecoveryKeyBagRef SOSRecoveryKeyBagCreateFromData(CFAllocatorRef allocator, CFDataRef data, CFErrorRef *error);
CFDataRef SOSRecoveryKeyBagGetKeyData(SOSRecoveryKeyBagRef rkbg, CFErrorRef *error);

bool SOSRecoveryKeyBagDSIDIs(SOSRecoveryKeyBagRef rkbg, CFStringRef dsid);

// Der encoding
const uint8_t* der_decode_RecoveryKeyBag(CFAllocatorRef allocator,
                                            SOSRecoveryKeyBagRef* RecoveryKeyBag, CFErrorRef *error,
                                            const uint8_t* der, const uint8_t *der_end);

size_t der_sizeof_RecoveryKeyBag(SOSRecoveryKeyBagRef RecoveryKeyBag, CFErrorRef *error);
uint8_t* der_encode_RecoveryKeyBag(SOSRecoveryKeyBagRef RecoveryKeyBag, CFErrorRef *error,
                                      const uint8_t *der, uint8_t *der_end);
#endif /* SOSRecoveryKeyBag_h */
