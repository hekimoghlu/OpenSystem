/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 2, 2021.
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
#import "WebPreferencesPrivate.h"

@interface WebPreferences (WebPreferencesInternal)

- (BOOL)_boolValueForKey:(NSString *)key;
- (void)_setBoolValue:(BOOL)value forKey:(NSString *)key;
- (int)_integerValueForKey:(NSString *)key;
- (void)_setIntegerValue:(int)value forKey:(NSString *)key;
- (float)_floatValueForKey:(NSString *)key;
- (void)_setFloatValue:(float)value forKey:(NSString *)key;
- (void)_setLongLongValue:(long long)value forKey:(NSString *)key;
- (long long)_longLongValueForKey:(NSString *)key;
- (void)_setUnsignedLongLongValue:(unsigned long long)value forKey:(NSString *)key;
- (unsigned long long)_unsignedLongLongValueForKey:(NSString *)key;
- (NSString *)_stringValueForKey:(NSString *)key;
- (void)_setStringValue:(NSString *)value forKey:(NSString *)key;

@property (nonatomic, setter=_setUseSystemAppearance:) BOOL _useSystemAppearance;

@end
