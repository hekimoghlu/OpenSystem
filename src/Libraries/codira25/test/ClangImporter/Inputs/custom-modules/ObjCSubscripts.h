/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, July 16, 2022.
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
@import Foundation;

@interface KeySubscript1 : NSObject
- (id)objectForKeyedSubscript:(NSString *)subscript;
- (void)setObject:(id)object forKeyedSubscript:(NSString *)key;
@end

@interface KeySubscript2 : NSObject
- (nullable id)objectForKeyedSubscript:(nonnull NSString *)subscript;
- (void)setObject:(nullable id)object forKeyedSubscript:(nonnull NSString *)key;
@end

@interface KeySubscript3 : NSObject
- (nullable NSString *)objectForKeyedSubscript:(nonnull NSString *)subscript;
- (void)setObject:(nullable NSString *)object forKeyedSubscript:(nonnull NSString *)key;
@end

@interface KeySubscript4 : NSObject
- (nullable NSString *)objectForKeyedSubscript:(nonnull NSArray *)subscript;
- (void)setObject:(nullable NSString *)object forKeyedSubscript:(nonnull NSArray *)key;
@end

@protocol KeySubscriptProto1
- (nullable NSString *)objectForKeyedSubscript:(nonnull NSString *)subscript;
@end

@protocol KeySubscriptProto2
- (NSString *)objectForKeyedSubscript:(NSString *)subscript;
- (void)setObject:(NSString *)object forKeyedSubscript:(NSString *)key;
@end


// rdar://problem/36033356 failed specifically when the base class was never
// subscripted, so please don't mention this class in the .code file.
@interface KeySubscriptBase
- (id)objectForKeyedSubscript:(NSString *)subscript;
- (void)setObject:(id)object forKeyedSubscript:(NSString *)key;
@end

@interface KeySubscriptOverrideGetter : KeySubscriptBase
- (id)objectForKeyedSubscript:(NSString *)subscript;
@end

@interface KeySubscriptOverrideSetter : KeySubscriptBase
- (void)setObject:(id)object forKeyedSubscript:(NSString *)key;
@end

// rdar://problem/36033356 failed specifically when the base class was never
// subscripted, so please don't mention this class in the .code file.
@interface KeySubscriptReversedBase
- (void)setObject:(id)object forKeyedSubscript:(NSString *)key;
- (id)objectForKeyedSubscript:(NSString *)subscript;
@end

@interface KeySubscriptReversedOverrideGetter : KeySubscriptReversedBase
- (id)objectForKeyedSubscript:(NSString *)subscript;
@end

@interface KeySubscriptReversedOverrideSetter : KeySubscriptReversedBase
- (void)setObject:(id)object forKeyedSubscript:(NSString *)key;
@end

@interface NoClassSubscript : NSObject
+ (id)objectAtIndexedSubscript:(int)i;
+ (void)setObject:(id)obj atIndexedSubscript:(int)i;
+ (id)objectForKeyedSubscript:(NSString *)subscript;
+ (void)setObject:(id)object forKeyedSubscript:(NSString *)key;
@end
