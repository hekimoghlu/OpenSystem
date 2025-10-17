/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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

//
//  _PMLowPowerModeProtocol.h
//
//  Created by Andrei Dorofeev on 1/14/15.
//  Copyright Â© 2015,2020 Apple Inc. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <LowPowerMode/_PMPowerModeState.h>

typedef void (^PMSetPowerModeCompletionHandler)(BOOL success, NSError *error);

@protocol _PMLowPowerModeProtocol

@optional
- (void)setPowerMode:(PMPowerMode)mode
          fromSource:(NSString *)source
      withCompletion:(PMSetPowerModeCompletionHandler)handler;

@required
- (void)setPowerMode:(PMPowerMode)mode
          fromSource:(NSString *)source
          withParams:(NSDictionary *)params // Session params applicable only while LPM is ON
      withCompletion:(PMSetPowerModeCompletionHandler)handler;

@end
