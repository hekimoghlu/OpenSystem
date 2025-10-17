/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 10, 2023.
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


// Make sure that we have set the -D flag appropriately.
#ifdef __LANGUAGE_ATTR_SUPPORTS_SENDING
#if !__LANGUAGE_ATTR_SUPPORTS_SENDING
#error "Compiler should have set __LANGUAGE_ATTR_SUPPORTS_SENDING to 1"
#endif
#else
#error "Compiler should have defined __LANGUAGE_ATTR_SUPPORTS_SENDING"
#endif

#define LANGUAGE_SENDING __attribute__((language_attr("sending")))

#pragma clang assume_nonnull begin

#ifdef __OBJC__

@import Foundation;

@interface MyType : NSObject
- (NSObject *)getSendingResult LANGUAGE_SENDING;
- (NSObject *)getSendingResultWithArgument:(NSObject *)arg LANGUAGE_SENDING;
- (NSObject *)getResultWithSendingArgument:(NSObject *)LANGUAGE_SENDING arg;
@end

LANGUAGE_SENDING
@interface DoesntMakeSense : NSObject
@end

NSObject *returnNSObjectFromGlobalFunction(NSObject *other);
NSObject *sendNSObjectFromGlobalFunction(NSObject *other) LANGUAGE_SENDING;
void sendNSObjectToGlobalFunction(NSObject *arg LANGUAGE_SENDING);

#endif

typedef struct {
  int state;
} NonSendableCStruct;

NonSendableCStruct
returnUserDefinedFromGlobalFunction(NonSendableCStruct other);
NonSendableCStruct
sendUserDefinedFromGlobalFunction(NonSendableCStruct other) LANGUAGE_SENDING;
void sendUserDefinedIntoGlobalFunction(
    NonSendableCStruct arg LANGUAGE_SENDING);

void sendingWithCompletionHandler(void (^completion)(LANGUAGE_SENDING NonSendableCStruct arg));
LANGUAGE_SENDING NonSendableCStruct sendingWithLazyReturn(LANGUAGE_SENDING NonSendableCStruct (^makeLazily)(void));

#pragma clang assume_nonnull end
