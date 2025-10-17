/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 11, 2022.
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
#ifndef mockaks_h
#define mockaks_h

#import <libaks.h>

#if __has_include(<MobileKeyBag/MobileKeyBag.h>)
#include <MobileKeyBag/MobileKeyBag.h>
#define HAVE_MobileKeyBag_MobileKeyBag 1
#endif

#if __OBJC2__
#import <Foundation/Foundation.h>
#endif

CF_ASSUME_NONNULL_BEGIN

#if !HAVE_MobileKeyBag_MobileKeyBag

typedef struct  __MKBKeyBagHandle* MKBKeyBagHandleRef;
int MKBKeyBagCreateWithData(CFDataRef keybagBlob, MKBKeyBagHandleRef _Nullable * _Nonnull newHandle);

#define kMobileKeyBagDeviceIsLocked 1
#define kMobileKeyBagDeviceIsUnlocked 0

int MKBKeyBagUnlock(MKBKeyBagHandleRef keybag, CFDataRef _Nullable passcode);
int MKBKeyBagGetAKSHandle(MKBKeyBagHandleRef _Nonnull keybag, int32_t *_Nullable handle);
int MKBGetDeviceLockState(CFDictionaryRef _Nullable options);
CF_RETURNS_RETAINED CFDictionaryRef _Nullable MKBUserTypeDeviceMode(CFDictionaryRef _Nullable options, CFErrorRef _Nullable * _Nullable error);
int MKBForegroundUserSessionID( CFErrorRef _Nullable * _Nullable error);

#define kMobileKeyBagSuccess (0)
#define kMobileKeyBagError (-1)
#define kMobileKeyBagDeviceLockedError (-2)
#define kMobileKeyBagInvalidSecretError (-3)
#define kMobileKeyBagExistsError                (-4)
#define kMobileKeyBagNoMemoryError    (-5)


#endif // HAVE_MobileKeyBag_MobileKeyBag

#if __OBJC2__

@interface SecMockAKS : NSObject
@property (class) keybag_state_t keybag_state;

+ (bool)isLocked:(keyclass_t)key_class;
+ (bool)forceInvalidPersona;
+ (bool)forceUnwrapKeyDecodeFailure;
+ (bool)isSEPDown;
+ (bool)useGenerationCount;

+ (void)lockClassA_C;
+ (void)lockClassA;

+ (void)unlockAllClasses;

+ (void)reset;

+ (void)failNextDecryptRefKey:(NSError* _Nonnull) decryptRefKeyError;
+ (void)resetDecryptRefKeyFailures;
+ (NSError * _Nullable)popDecryptRefKeyFailure;

+ (void)setOperationsUntilUnlock:(int)val;

@end

#endif // OBJC2

CF_ASSUME_NONNULL_END


#endif /* mockaks_h */
