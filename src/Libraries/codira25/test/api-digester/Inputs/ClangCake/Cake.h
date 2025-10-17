/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 16, 2021.
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

#import <objc_generics.h>
extern int ANTGlobalValue;

@interface NewType
@end
@interface OldType
@end

@protocol TypeWithMethod
  -(void) minusPrint;
  +(void) plusPrint;
  @property int PropertyA;
@end

@protocol ObjcProt
  -(void) ProtMemberFunc;
@end

typedef NSString * AnimalAttributeName NS_STRING_ENUM;

@interface AnimalStatusDescriptor
- (nonnull AnimalStatusDescriptor *)animalStatusDescriptorByAddingAttributes:(nonnull NSDictionary<AnimalAttributeName, id> *)attributes;
- (nonnull AnimalStatusDescriptor *)animalStatusDescriptorByAddingOptionalAttributes:(nullable NSDictionary<AnimalAttributeName, id> *)attributes;
- (nonnull AnimalStatusDescriptor *)animalStatusDescriptorByAddingAttributesArray:(nonnull NSArray<AnimalAttributeName> *)attributes;
- (nonnull AnimalStatusDescriptor *)animalStatusDescriptorByAddingOptionalAttributesArray:(nullable NSArray<AnimalAttributeName> *)attributes;
+ (nonnull AnimalStatusDescriptor *)animalStatusSingleOptionalAttribute:(nullable AnimalAttributeName)attributes;
+ (nonnull AnimalStatusDescriptor *)animalStatusSingleAttribute:(nonnull AnimalAttributeName)attributes;
@end

extern AnimalAttributeName globalAttributeName;

typedef NSString * CatAttributeName NS_STRING_ENUM;
