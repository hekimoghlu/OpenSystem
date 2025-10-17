/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 9, 2023.
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
//  _PMCoreSmartPowerNapCallbackProtocol.h
//  LowPowerMode-Embedded
//
//  Created by Prateek Malhotra on 12/7/22.
//

#ifndef _PMCoreSmartPowerNapCallbackProtocol_h
#define _PMCoreSmartPowerNapCallbackProtocol_h

#import <LowPowerMode/_PMCoreSmartPowerNapProtocol.h>

@protocol _PMCoreSmartPowerNapCallbackProtocol

// callback receieved from powerd when state changes
- (void)updateState: (_PMCoreSmartPowerNapState)state;

@end


#endif /* _PMCoreSmartPowerNapCallbackProtocol_h */
