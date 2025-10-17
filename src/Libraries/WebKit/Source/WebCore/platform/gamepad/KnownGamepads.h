/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 12, 2024.
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
#pragma once

#if ENABLE(GAMEPAD)

namespace WebCore {

enum KnownGamepad {
    Dualshock3 = 0x054c0268,
    Dualshock4_1 = 0x054c05c4,
    Dualshock4_2 = 0x054c09cc,
    GamesirM2 = 0x0ec20475,
    HoripadUltimate = 0x0f0d0090,
    LogitechF310 = 0x046dc216,
    LogitechF710 = 0x046dc219,
    Nimbus1 = 0x01111420,
    Nimbus2 = 0x10381420,
    StadiaA = 0x18d19400,
    StratusXL1 = 0x01111418,
    StratusXL2 = 0x10381418,
    StratusXL3 = 0x01111419,
    StratusXL4 = 0x10381419,
    XboxOne1 = 0x045e02e0,
    XboxOne2 = 0x045e02ea,
    XboxOne3 = 0x045e02fd,
};

} // namespace WebCore

#endif // ENABLE(GAMEPAD)
