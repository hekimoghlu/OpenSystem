/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 16, 2024.
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
//  _PMPowerModes.h
//
//  Created by Prateek Malhotra on 6/24/24.
//  Copyright Â© 2024 Apple Inc. All rights reserved.
//

#import "_PMPowerModesProtocol.h"

#if TARGET_OS_OSX

@interface _PMPowerModes : NSObject <_PMPowerModesProtocol>

+ (instancetype)sharedInstance;

/**
 * @abstract            Fetch information about the currently active power mode session
 * @return              The session info object
 * */
- (_PMPowerModeSession *)currentPowerModeSession;

/**
 * @abstract            The currently active power mode
 * */
- (PMPowerMode)currentPowerMode;

/**
 Whether power modes can be controlled through the UI.
 */
/**
 * @abstract            Whether power modes can be controlled through the UI.
 * @discussion          On non-Battery Macs, modes may not be available for selection through ControlCenter but only be available through Settings.
 * @return              YES, if mode selection is supported through the UI (Control Center and/or MenuExtra)
 * */

- (BOOL)supportsPowerModeSelectionUI;

@end

#endif
