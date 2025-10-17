/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 11, 2022.
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

#include "MediaCapabilitiesInfo.h"
#include "MediaEncodingConfiguration.h"

namespace WebCore {

struct MediaCapabilitiesEncodingInfo : MediaCapabilitiesInfo {
    // FIXME(C++17): remove the following constructors once all compilers support extended
    // aggregate initialization:
    // <http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/p0017r1.html>
    MediaCapabilitiesEncodingInfo() = default;
    MediaCapabilitiesEncodingInfo(MediaEncodingConfiguration&& supportedConfiguration)
        : MediaCapabilitiesEncodingInfo({ }, WTFMove(supportedConfiguration))
    {
    }
    MediaCapabilitiesEncodingInfo(MediaCapabilitiesInfo&& info, MediaEncodingConfiguration&& supportedConfiguration)
        : MediaCapabilitiesInfo(WTFMove(info))
        , supportedConfiguration(WTFMove(supportedConfiguration))
    {
    }

    MediaEncodingConfiguration supportedConfiguration;

    MediaCapabilitiesEncodingInfo isolatedCopy() const;

};

inline MediaCapabilitiesEncodingInfo MediaCapabilitiesEncodingInfo::isolatedCopy() const
{
    return { MediaCapabilitiesInfo::isolatedCopy(), supportedConfiguration.isolatedCopy() };
}

} // namespace WebCore

