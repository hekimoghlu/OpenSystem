/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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

@import ObjectiveC;

@interface NSString : NSObject
- (NSString*)uppercaseString;
@end

@interface NSMutableString : NSString
@end

@interface NSArray<ObjectType> : NSObject
@end

@interface NSMutableArray<ObjectType> : NSArray<ObjectType>
@end

@interface NSDictionary<KeyType, ValueType> : NSObject
@end

@interface NSSet<ObjectType> : NSObject
@end

@interface NSMutableSet<ObjectType> : NSSet<ObjectType>
@end

@interface NSNumber : NSObject
@end

@interface NSNotification : NSObject
@end

@interface Foo

- (NSString*) foo;
- (void) setFoo: (NSString*)s;

@end

NSString *bar(int);
void setBar(NSString *s);

#define CF_ENUM(_type, _name) enum _name : _type _name; enum _name : _type
#define NS_ENUM(_type, _name) CF_ENUM(_type, _name)

@interface NSManagedObject: NSObject
@end

@interface NSData: NSObject <NSCopying>
@end

typedef struct __CGImage *CGImageRef;

__attribute__((availability(macosx,introduced=10.51)))
@interface NSUserNotificationAction : NSObject
@end

void always_available_function();

__attribute__((availability(macosx,introduced=10.51)))
void future_function_should_be_weak();

extern int weak_variable __attribute__((weak_import));
extern int strong_variable;

@interface NSError : NSObject

@property NSInteger code;
@property NSString *domain;
@property NSDictionary *userInfo;

@end

typedef NSString *_Nonnull NSNotificationName
    __attribute((language_newtype(struct)));
