/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 17, 2025.
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
#include "config.h"
#include "ProcessCapabilities.h"

namespace WebCore {

#if USE(CG)
static bool s_HEICDecodingEnabled = false;
static bool s_AVIFDecodingEnabled = false;
#endif

static bool s_hardwareAcceleratedDecodingDisabled = false;
static bool s_canUseAcceleratedBuffers = true;

#if USE(CG)
void ProcessCapabilities::setHEICDecodingEnabled(bool value)
{
    s_HEICDecodingEnabled = value;
}

bool ProcessCapabilities::isHEICDecodingEnabled()
{
    return s_HEICDecodingEnabled;
}

void ProcessCapabilities::setAVIFDecodingEnabled(bool value)
{
    s_AVIFDecodingEnabled = value;
}

bool ProcessCapabilities::isAVIFDecodingEnabled()
{
    return s_AVIFDecodingEnabled;
}
#endif

void ProcessCapabilities::setHardwareAcceleratedDecodingDisabled(bool value)
{
    s_hardwareAcceleratedDecodingDisabled = value;
}

bool ProcessCapabilities::isHardwareAcceleratedDecodingDisabled()
{
    return s_hardwareAcceleratedDecodingDisabled;
}

void ProcessCapabilities::setCanUseAcceleratedBuffers(bool value)
{
    s_canUseAcceleratedBuffers = value;
}

bool ProcessCapabilities::canUseAcceleratedBuffers()
{
    return s_canUseAcceleratedBuffers;
}

} // namespace WebCore
