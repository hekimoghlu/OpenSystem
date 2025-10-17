/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, April 15, 2023.
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
#ifndef TokenLogin_h
#define TokenLogin_h

#include <CoreFoundation/CoreFoundation.h>

#ifdef __cplusplus
extern "C" {
#endif

OSStatus TokenLoginGetContext(const void *base64TokenLoginData, UInt32 base64TokenLoginDataLength, CFDictionaryRef *context);
OSStatus TokenLoginGetLoginData(CFDictionaryRef context, CFDictionaryRef *loginData);
OSStatus TokenLoginGetPin(CFDictionaryRef context, CFStringRef *pin);

OSStatus TokenLoginCreateLoginData(CFStringRef tokenId, CFDataRef pubKeyHash, CFDataRef pubKeyHashWrap, CFDataRef unlockKey, CFDataRef scBlob);
OSStatus TokenLoginUpdateUnlockData(CFDictionaryRef context, CFStringRef password);
OSStatus TokenLoginStoreUnlockData(CFDictionaryRef context, CFDictionaryRef loginData);
OSStatus TokenLoginDeleteUnlockData(CFDataRef pubKeyHash);

OSStatus TokenLoginGetUnlockKey(CFDictionaryRef context, CFDataRef *unlockKey);
OSStatus TokenLoginGetScBlob(CFDataRef pubKeyHash, CFStringRef tokenId, CFStringRef password, CFDataRef *scBlob);
OSStatus TokenLoginUnlockKeybag(CFDictionaryRef context, CFDictionaryRef loginData);
CFStringRef TokenLoginCopyKcPwd(CFDictionaryRef context);
CFStringRef TokenLoginCopyUserPwd(CFDictionaryRef context);

#ifdef __cplusplus
}
#endif

#endif /* TokenLogin_h */
