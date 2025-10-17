/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 15, 2024.
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
#if USE(APPLE_INTERNAL_SDK)

#import <PrototypeTools/PrototypeTools.h>

#else

@interface PTSettings : NSObject <NSCopying>
- (void)setDefaultValues;
@end

@interface PTDomain : NSObject
+ (__kindof PTSettings *)rootSettings;
@end

@interface PTSection : NSObject
@end

@interface PTModule : NSObject
+ (instancetype)moduleWithTitle:(NSString *)title contents:(NSArray *)contents;
+ (PTSection *)sectionWithRows:(NSArray *)rows title:(NSString *)title;
+ (PTSection *)sectionWithRows:(NSArray *)rows;
@end

@interface PTRow : NSObject <NSCopying, NSSecureCoding>
+ (instancetype)rowWithTitle:(NSString *)staticTitle valueKeyPath:(NSString *)keyPath;
- (id)valueValidator:(id(^)(id proposedValue, id settings))validator;
- (id)condition:(NSPredicate *)condition;
@end

@interface PTSRow : PTRow
@end

@interface PTSwitchRow : PTSRow
@end

@interface PTSliderRow : PTSRow
- (id)minValue:(CGFloat)minValue maxValue:(CGFloat)maxValue;
@end

@interface PTEditFloatRow : PTSRow
@end

@interface PTRowAction : NSObject
@end

@interface PTRestoreDefaultSettingsRowAction : PTRowAction
+ (instancetype)action;
@end

@interface PTButtonRow : PTSRow
+ (instancetype)rowWithTitle:(NSString *)staticTitle action:(PTRowAction *)action;
@end

#endif
