/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, May 19, 2025.
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

#import "objc_direct.h"

@implementation Bar
- (instancetype)initWithValue:(int)value {
  printf("called %s with %d\n", __FUNCTION__, value);
  self = [super init];
  _directProperty = value;
  return self;
}
- (int)objectAtIndexedSubscript:(int)i {
  return 789;
}
- (void)setObject:(int)obj atIndexedSubscript:(int)i {}
- (NSString *)directMethod {
  return @"called directMethod";
}
+ (NSString *)directClassMethod {
  return @"called directClassMethod";
}
- (NSString *)directProtocolMethod {
  return @"called directProtocolMethod";
}
@end

@implementation Bar(CategoryName)
- (int)directProperty2 {
  return 456;
}
- (void)setDirectProperty2:(int)i {}
- (NSString *)directMethod2 {
  return @"called directMethod2";
}
+ (NSString *)directClassMethod2 {
  return @"called directClassMethod2";
}
@end
