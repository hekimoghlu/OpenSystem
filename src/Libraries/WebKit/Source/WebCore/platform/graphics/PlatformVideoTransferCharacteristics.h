/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 20, 2022.
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

#include <wtf/Forward.h>

namespace WebCore {

enum class PlatformVideoTransferCharacteristics : uint8_t {
    Bt709,
    Smpte170m,
    Iec6196621,
    Gamma22curve,
    Gamma28curve,
    Smpte240m,
    Linear,
    Log,
    LogSqrt,
    Iec6196624,
    Bt1361ExtendedColourGamut,
    Bt2020_10bit,
    Bt2020_12bit,
    SmpteSt2084,
    SmpteSt4281,
    AribStdB67Hlg,
    Unspecified,

    // Aliases for WebIDL bindings
    PQ = SmpteSt2084,
    HLG = AribStdB67Hlg,
};

} // namespace WebCore
