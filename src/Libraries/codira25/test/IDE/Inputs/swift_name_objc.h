/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 23, 2024.
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

@import Foundation;

#define LANGUAGE_NAME(X) __attribute__((language_name(#X)))

// Renaming classes
LANGUAGE_NAME(SomeClass)
@interface SNSomeClass : NSObject
- (instancetype)initWithFloat:(float)f;
- (instancetype)initWithDefault;
- (void)instanceMethodWithX:(float)x Y:(float)y Z:(float)z;
+ (instancetype)someClassWithDouble:(double)d;
+ (instancetype)someClassWithTry:(BOOL)shouldTry;
+ (NSObject *)buildWithObject:(NSObject *)object LANGUAGE_NAME(init(object:));
+ (instancetype)buildWithUnsignedChar:(unsigned char)uint8 LANGUAGE_NAME(init(uint8:));
@property (readonly,nonatomic) float floatProperty;
@property (readwrite,nonatomic) double doubleProperty;
@end

LANGUAGE_NAME(SomeProtocol)
@protocol SNSomeProtocol
- (void)protoInstanceMethodWithX:(float)x y:(float)y;
@end

@interface SNSomeClass ()
- (void)extensionMethodWithX:(float)x y:(float)y;
@end

@interface SNSomeClass (Category1) <SNSomeProtocol>
- (void)categoryMethodWithX:(float)x y:(float)y;
- (void)categoryMethodWithX:(float)x y:(float)y z:(float)z;
- (id)objectAtIndexedSubscript:(NSInteger)index;
@end

@interface SNCollision
@end

@protocol SNCollision
@property (readonly,nonnull) id reqSetter;
- (void)setReqSetter:(nonnull id)bar;

@property (readonly,nonnull) id optSetter;
@optional
- (void)setOptSetter:(nonnull id)bar;
@end

@protocol NSAccessibility
@property (nonatomic) float accessibilityFloat;
@end

@interface UIActionSheet : NSObject
-(instancetype)initWithTitle:(const char *)title delegate:(id)delegate cancelButtonTitle:(const char *)cancelButtonTitle destructiveButtonTitle:(const char *)destructiveButtonTitle otherButtonTitles:(const char *)otherButtonTitles, ...;
@end

@interface NSErrorImports : NSObject
- (nullable NSObject *)methodAndReturnError:(NSError **)error;
- (BOOL)methodWithFloat:(float)value error:(NSError **)error;
- (nullable void *)pointerMethodAndReturnError:(NSError **)error;
- (nullable SEL)selectorMethodAndReturnError:(NSError **)error;
- (nullable void (*)(void))functionPointerMethodAndReturnError:(NSError **)error;
- (nullable void (^)(void))blockMethodAndReturnError:(NSError **)error;

- (nullable instancetype)initAndReturnError:(NSError **)error;
- (nullable instancetype)initWithFloat:(float)value error:(NSError **)error;

- (nonnull void *)badPointerMethodAndReturnError:(NSError **)error;
@end

typedef const void *CFTypeRef __attribute__((objc_bridge(id)));
typedef const struct __attribute__((objc_bridge(id))) __CCItem *CCItemRef;
