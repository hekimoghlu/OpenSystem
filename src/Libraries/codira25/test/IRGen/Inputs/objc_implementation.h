/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 27, 2025.
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

#if __OBJC__
#import <Foundation/Foundation.h>

@interface ImplClass: NSObject <NSCopying>

- (nonnull instancetype)init;

@property (assign) int implProperty;

- (void)mainMethod:(int)param;

- (void)asyncMethodWithCompletionHandler:(void (^ _Nullable)(void))completion;

@end

@interface ImplClass () <NSMutableCopying>

- (void)extensionMethod:(int)param;

@end


@interface ImplClass (Category1)

- (void)category1Method:(int)param;

@end


@interface ImplClass (Category2)

- (void)category2Method:(int)param;

@end
#endif

extern void implFunc(int param);
extern void implFuncCName(int param) __asm__("_implFuncAsmName");
extern void implFuncRenamed_C(int param) __attribute__((language_name("implFuncRenamed_Codira(param:)")));

#if __OBJC__
@interface NoImplClass

- (void)noImplMethod:(int)param;

@end

@interface NoInitImplClass: NSObject

@property (readonly, strong, nonnull) NSString *s1;
@property (strong, nonnull) NSString *s2;
@property (readonly, strong, nonnull) NSString *s3;
@property (strong, nonnull) NSString *s4;

@end

@interface ImplClassWithResilientStoredProperty : NSObject

@property (assign) int beforeInt;
@property (assign) int afterInt;

@end
#endif
