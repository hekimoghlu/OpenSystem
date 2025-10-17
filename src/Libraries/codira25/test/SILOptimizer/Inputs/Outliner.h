/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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

#import <Foundation/Foundation.h>

@protocol Proto
- (id)requirement;
@end

@interface Gizmo : NSObject
@property (nonatomic)NSString *stringProperty;
- (NSString*) modifyString: (NSString *)str withNumber: (NSInteger) num withFoobar: (id)foobar;
- (id) doSomething : (NSArray<NSString*>*) arr;
@end

@interface Gizmo2<ObjectType: id<Proto>> : NSObject
- (NSString*) doSomething;
@end

@protocol FooProto <NSObject>
@end

@protocol SomeGenericClass <FooProto>
@property (nonatomic, nullable, readonly, strong) NSString *version;
- (NSString*) doSomething;
- (id) doSomething2 : (NSArray<NSString*>*) arr;
@end

typedef NS_ENUM(NSUInteger, MyEventType) {
    MyEventTypeA = 1,
    MyEventTypeB = 2
};

@interface MyWindow : NSObject
@property NSInteger windowNumber;
@end

@interface MyView : NSObject
@property (nonatomic, nullable, readonly, strong) MyWindow *window;
@property (nonatomic, nullable, strong) MyWindow *window2;
@end

typedef struct MyPoint {
NSInteger x;
NSInteger y;
} MyPoint;

@interface MyGraphicsContext : NSObject
@end

@interface MyEvent : NSObject
+ (nullable MyEvent *)mouseEventWithType:(MyEventType)type
                                location:(MyPoint)pt
                            windowNumber:(NSInteger)wNum
                                 context:(nullable MyGraphicsContext * __unused)context
                             eventNumber:(NSInteger)eNum
                              clickCount:(NSInteger)cnt
                                pressure:(float)pressure;
@end

NS_ASSUME_NONNULL_BEGIN
@protocol Treeish <NSObject>
- (nullable NSArray *) treeishChildren;
@end
NS_ASSUME_NONNULL_END

NS_ASSUME_NONNULL_BEGIN
@interface MyObject : NSObject
@property (nullable) NSError *error;
@end
NS_ASSUME_NONNULL_END
