/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 20, 2024.
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

struct OptionsStruct {
  int intOption;
  float floatOption;
};

@interface OptionsConsumerObjC : NSObject

- (nonnull instancetype)initWithOptions:(const OptionsStruct &)options;
+ (nonnull instancetype)consumerWithOptions:(const OptionsStruct &)options;
+ (int)doThingWithOptions:(const OptionsStruct &)options;
- (float)doOtherThingWithOptions:(const OptionsStruct &)options;

@end

struct OptionsConsumerCxx {
  OptionsConsumerCxx(const OptionsStruct &options);
  static OptionsConsumerCxx build(const OptionsStruct &options);
  static int doThing(const OptionsStruct &options);
  float doOtherThing(const OptionsStruct &options);
};
