/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 23, 2023.
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

#import "ObjectiveC.h"
#import "dispatch.h"

#define NS_OPTIONS(type, name) type name; enum

#pragma clang assume_nonnull begin

@interface NSString : NSObject<NSCopying>

- (null_unspecified NSString*)uppercaseString;
- (id) copyWithZone: (nullable void*)zone;

@end

typedef NSString *NSErrorDomain;

@interface NSArray<ObjectType> : NSObject

- (_Nonnull instancetype)initWithObjects:(const _Null_unspecified ObjectType * _Null_unspecified)objects count:(int)count;
- (_Nonnull instancetype)initWithArray:(NSArray<ObjectType> *_Null_unspecified)array;

- (nonnull ObjectType)objectAtIndexedSubscript:(NSInteger)i;

@end

@interface NSDictionary<KeyType, ValueType> : NSObject

+ (void) classMethod;

@end

@interface NSSet<ObjectType> : NSObject
@end

@interface NSNumber : NSObject
@end

@interface Foo : NSObject

- (NSString * _Null_unspecified ) foo;
- (void)setFoo:(NSString * _Null_unspecified)s;

- (BOOL) zim;
- (void) setZim: (BOOL)b;

- (_Bool) zang;
- (void) setZang: (_Bool)b;

@property int intProperty;

@end

@interface NSError : NSObject

@property NSInteger code;
@property NSString *domain;
@property NSDictionary *userInfo;

@end

@interface NSDraggingItem

@property(copy, nullable) NSArray *_Nonnull (^imageComponentsProvider)(void);

@end

NSString * _Null_unspecified bar(void);
void setBar(NSString * _Null_unspecified s);

NSString * _Null_unspecified NSStringFromString(NSString * _Null_unspecified s);
NSString *NSStringFromClass(Class c);

#define CF_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NS_ENUM(_type, _name) CF_ENUM(_type, _name)

BOOL getBOOL(void);
_Bool getBool(void);


void useBOOL(BOOL x);
void useBool(_Bool x);

#pragma clang assume_nonnull end

void nonnullStringBlockResult(NSString *_Nonnull (^ _Null_unspecified block)(void));
void nonnullArrayBlockResult(NSArray *_Nonnull (^ _Null_unspecified block)(void));
void nonnullDictionaryBlockResult(NSDictionary *_Nonnull (^ _Null_unspecified block)(void));
void nonnullSetBlockResult(NSSet *_Nonnull (^ _Null_unspecified block)(void));

void noescapeBlock(__attribute__((noescape)) void (^ _Null_unspecified block)(void));
void escapeBlock(void (^ _Null_unspecified block)(void));
void noescapeBlock3(__attribute__((noescape)) void (^ _Null_unspecified block)(NSString * _Null_unspecified s),
                    __attribute__((noescape)) void (^ _Null_unspecified block2)(NSString * _Null_unspecified s),
                    NSString * _Null_unspecified f);

void noescapeNonnullBlock(__attribute__((noescape)) void (^_Nonnull block)(void));
void escapeNonnullBlock(void (^_Nonnull block)(void));

void noescapeBlockAlias(__attribute__((noescape)) _Null_unspecified dispatch_block_t block);
void noescapeNonnullBlockAlias(__attribute__((noescape)) _Nonnull dispatch_block_t block);
void escapeBlockAlias(_Null_unspecified dispatch_block_t block);

@interface ObjectWithSplitProperty : NSObject
@property (nonatomic, setter=private_setFlagForSomething:) BOOL flagForSomething;
@end

extern NSString * _Nonnull (^ const _Nonnull GlobalBlock)(NSString * _Nonnull);
