/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 12, 2023.
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

#include <optional>
#include <wtf/Forward.h>
#include <wtf/Vector.h>

namespace WebCore {

class SharedBuffer;
struct FourCC;

struct AVCParameters {
    uint8_t profileIDC { 0 };
    uint8_t constraintsFlags { 0 };
    uint8_t levelIDC { 0 };
};

WEBCORE_EXPORT std::optional<AVCParameters> parseAVCCodecParameters(StringView);
WEBCORE_EXPORT std::optional<AVCParameters> parseAVCDecoderConfigurationRecord(const SharedBuffer&);
WEBCORE_EXPORT String createAVCCodecParametersString(const AVCParameters&);

struct HEVCParameters {
    enum class Codec { Hev1, Hvc1 } codec { Codec::Hvc1 };
    uint16_t generalProfileSpace { 0 };
    uint16_t generalProfileIDC { 0 };
    uint32_t generalProfileCompatibilityFlags { 0 };
    uint8_t generalTierFlag { 0 };
    Vector<unsigned char, 6> generalConstraintIndicatorFlags { 0, 0, 0, 0, 0, 0 };
    uint16_t generalLevelIDC { 0 };
};

WEBCORE_EXPORT std::optional<HEVCParameters> parseHEVCCodecParameters(StringView);
WEBCORE_EXPORT std::optional<HEVCParameters> parseHEVCDecoderConfigurationRecord(FourCC, const SharedBuffer&);
WEBCORE_EXPORT String createHEVCCodecParametersString(const HEVCParameters&);

struct DoViParameters {
    enum class Codec { AVC1, AVC3, HEV1, HVC1 } codec { Codec::HVC1 };
    uint16_t bitstreamProfileID { 0 };
    uint16_t bitstreamLevelID { 0 };
};

WEBCORE_EXPORT std::optional<DoViParameters> parseDoViCodecParameters(StringView);
WEBCORE_EXPORT std::optional<DoViParameters> parseDoViDecoderConfigurationRecord(const SharedBuffer&);
WEBCORE_EXPORT String createDoViCodecParametersString(const DoViParameters&);

}
