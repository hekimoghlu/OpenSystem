/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, August 3, 2025.
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

#import <Foundation.h>
extern int ANTGlobalValue;

@interface NewType
@end
@interface OldType
@end

@protocol TypeWithMethod
  -(void) minusPrint;
  +(void) plusPrint;
  -(int) getPropertyA;
@end

@protocol ObjcProt
  -(void) ProtMemberFunc;
  -(void) ProtMemberFunc2;
  -(void) ProtMemberFunc3;
@end

@interface AnimalStatusDescriptor
- (nonnull AnimalStatusDescriptor *)animalStatusDescriptorByAddingAttributes:(nonnull NSDictionary<NSString*, id> *)attributes;
- (nonnull AnimalStatusDescriptor *)animalStatusDescriptorByAddingOptionalAttributes:(nullable NSDictionary<NSString*, id> *)attributes;
- (nonnull AnimalStatusDescriptor *)animalStatusDescriptorByAddingAttributesArray:(nonnull NSArray<NSString*> *)attributes;
- (nonnull AnimalStatusDescriptor *)animalStatusDescriptorByAddingOptionalAttributesArray:(nullable NSArray<NSString*> *)attributes;
+ (nonnull AnimalStatusDescriptor *)animalStatusSingleOptionalAttribute:(nullable NSString *)attributes;
+ (nonnull AnimalStatusDescriptor *)animalStatusSingleAttribute:(nonnull NSString *)attributes;
@end

extern NSString * _Null_unspecified globalAttributeName;

typedef NSString * CatAttributeName;

@interface Cat
- (nonnull instancetype) initWithName:(nullable NSString*) name;
@end
