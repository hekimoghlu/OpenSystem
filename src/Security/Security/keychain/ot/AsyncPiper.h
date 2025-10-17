/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 30, 2024.
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
#if __OBJC2__

#ifndef AyncPiper_h
#define AsyncPiper_h

NS_ASSUME_NONNULL_BEGIN

@interface AsyncPiper : NSObject

+ (instancetype)new NS_UNAVAILABLE;
- (instancetype)init NS_UNAVAILABLE;

-(instancetype)initWithError:(NSError** _Nullable)error NS_DESIGNATED_INITIALIZER;
-(xpc_object_t)xpcFd;
-(void)waitAndReleaseFd_ForTestingOnly;
-(NSDictionary*)dictWithError:(NSError**)errorOut;

@end

// Helper for later macros
#define AsyncPiperForTestingFailHelper(X) _AsyncPiperForTestingFail##X * failWrapping __attribute__((objc_precise_lifetime)) = [[_AsyncPiperForTestingFail##X alloc] init]

// Use this macro to force `pipe` in [AsyncPiper init] to fail for the scope in which the macro is invoked
#define AsyncPiperForTestingFailPipe AsyncPiperForTestingFailHelper(Pipe)

@interface _AsyncPiperForTestingFailPipe : NSObject

-(instancetype)init;

@end

// Use this macro to force `xpc_fd_create` in [AsyncPiper init] to fail for the scope in which the macro is invoked
#define AsyncPiperForTestingFailXpcFdWrapping AsyncPiperForTestingFailHelper(XpcFdWrapping)

@interface _AsyncPiperForTestingFailXpcFdWrapping : NSObject

-(instancetype)init;

@end

#endif /* AsyncPiper_h */

NS_ASSUME_NONNULL_END

#endif // ___OBJC2__
