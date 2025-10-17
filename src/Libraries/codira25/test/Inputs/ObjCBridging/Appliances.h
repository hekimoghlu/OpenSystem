/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 15, 2025.
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

#if __has_feature(objc_modules)
@import Foundation;
#else
#import <Foundation/Foundation.h>
#endif

@interface APPRefrigerator : NSObject <NSCopying>
-(_Nonnull instancetype)initWithTemperature:(double)temperature __attribute__((objc_designated_initializer));
@property (nonatomic) double temperature;
@end

@interface APPHouse : NSObject
@property (nonatomic,nonnull,copy) APPRefrigerator *fridge;
@end


@interface APPManufacturerInfo <DataType> : NSObject
@property (nonatomic,nonnull,readonly) DataType value;
@end

@interface APPBroken : NSObject
@property (nonatomic,nonnull,readonly) id thing;
@end

void takesNonStandardBlock(__attribute__((__ns_returns_retained__)) _Null_unspecified id (^ _Null_unspecified)(void));
