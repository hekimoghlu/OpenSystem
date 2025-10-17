/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, October 5, 2021.
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
#if USE(APPKIT)

#if USE(APPLE_INTERNAL_SDK)

#import <AppKit/NSAccessibilityRemoteUIElement.h>

#else

@interface NSAccessibilityRemoteUIElement : NSObject

+ (BOOL)isRemoteUIApp;
+ (void)setRemoteUIApp:(BOOL)flag;
+ (NSData *)remoteTokenForLocalUIElement:(id)localUIElement;
+ (void)registerRemoteUIProcessIdentifier:(pid_t)pid;
+ (void)unregisterRemoteUIProcessIdentifier:(pid_t)pid;

- (id)initWithRemoteToken:(NSData *)remoteToken;
- (pid_t)processIdentifier;
- (void)accessibilitySetPresenterProcessIdentifier:(pid_t)presenterPID;

@property (retain) id windowUIElement;
@property (retain) id topLevelUIElement;

@end

#endif // USE(APPLE_INTERNAL_SDK)

WTF_EXTERN_C_BEGIN

extern NSString *const NSApplicationDidChangeAccessibilityEnhancedUserInterfaceNotification;

void NSAccessibilityHandleFocusChanged();
void NSAccessibilityUnregisterUniqueIdForUIElement(id element);

WTF_EXTERN_C_END

#elif PLATFORM(MACCATALYST)

@interface NSObject (NSAccessibilityRemoteUIElement_Private)

- (void)registerRemoteUIProcessIdentifier:(pid_t)pid;
- (void)unregisterRemoteUIProcessIdentifier:(pid_t)pid;

@end

#endif // USE(APPKIT)
