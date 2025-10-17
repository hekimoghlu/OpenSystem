/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 2, 2024.
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
//  _PMPowerModeState.h
//
//  Created by Prateek Malhotra on 6/25/24.
//  Copyright Â© 2024 Apple Inc. All rights reserved.
//

typedef NS_ENUM(NSInteger, PMPowerMode) {
    PMNormalPowerMode = 0,
    PMLowPowerMode = 1,
#if TARGET_OS_OSX
    PMHighPowerMode = 2,
#endif
};

typedef NS_ENUM(NSInteger, PMPowerModeState) {
    PMPowerModeStateOff = 0,
    PMPowerModeStateOn = 255
};
