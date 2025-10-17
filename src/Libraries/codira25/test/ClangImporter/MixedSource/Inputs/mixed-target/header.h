/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 16, 2022.
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

// Don't change this to @import; it tickles a particular former crash.
#import <Foundation.h>

@import ExternIntX;
#import "Protocols.h"

#import "used-by-both-headers.h"

@class ForwardClass;
@protocol ForwardProto;

void doSomething(ForwardClass *arg);
void doSomethingProto(id <ForwardProto> arg);

@interface Base
- (NSObject *)safeOverride:(ForwardClass *)arg;
- (NSObject *)unsafeOverrideParam:(NSObject *)arg;
- (ForwardClass *)unsafeOverrideReturn:(ForwardClass *)arg;
@end

@protocol ForwardClassUser
- (void)consumeForwardClass:(ForwardClass *)arg;
@property ForwardClass *forward;
@end


@interface Base ()
- (NSObject *)safeOverrideProto:(id <ForwardProto>)arg;
- (NSObject *)unsafeOverrideProtoParam:(NSObject *)arg;
- (id <ForwardProto>)unsafeOverrideProtoReturn:(id <ForwardProto>)arg;
@end


@class PartialBaseClass;
@class PartialSubClass /* : NSObject */;
void doSomethingPartialBase(PartialBaseClass *arg);
void doSomethingPartialSub(PartialSubClass *arg);

@interface Base ()
- (NSObject *)safeOverridePartialSub:(PartialSubClass *)arg;
- (NSObject *)unsafeOverridePartialSubParam:(NSObject *)arg;
- (PartialSubClass *)unsafeOverridePartialSubReturn:(PartialSubClass *)arg;
@end


typedef NS_ENUM(short, AALevel) {
  AAA = 1,
  BBB = 2
};


@interface ConflictingName1
@end
@protocol ConflictingName1
@end
@protocol ConflictingName2
@end
@interface ConflictingName2
@end

@interface WrapperInterface
typedef int NameInInterface;
@end

@protocol WrapperProto
typedef int NameInProtocol;
@end

@interface WrapperInterface (Category)
typedef int NameInCategory;
@end


@protocol ForwardProtoFromOtherFile;
@interface ClassThatHasAProtocolTypedPropertyButMembersAreNeverLoaded
@property (weak) id <ForwardProtoFromOtherFile> weakProtoProp;
@end


@interface GenericObjCClass<Param : id <ForwardProto>> : Base
- (instancetype)init;
@end
