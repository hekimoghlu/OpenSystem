/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 11, 2022.
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

#include "PlatformVideoColorSpace.h"
#include "ScreenDataOverrides.h"
#include <wtf/text/StringView.h>
#include <wtf/text/WTFString.h>
namespace WebCore {

namespace VPConfigurationLevel {
constexpr uint8_t Level_1   = 10;
constexpr uint8_t Level_1_1 = 11;
constexpr uint8_t Level_2   = 20;
constexpr uint8_t Level_2_1 = 21;
constexpr uint8_t Level_3   = 30;
constexpr uint8_t Level_3_1 = 31;
constexpr uint8_t Level_4   = 40;
constexpr uint8_t Level_4_1 = 41;
constexpr uint8_t Level_5   = 50;
constexpr uint8_t Level_5_1 = 51;
constexpr uint8_t Level_5_2 = 52;
constexpr uint8_t Level_6   = 60;
constexpr uint8_t Level_6_1 = 61;
constexpr uint8_t Level_6_2 = 62;
}

namespace VPConfigurationChromaSubsampling {
constexpr uint8_t Subsampling_420_Vertical = 0;
constexpr uint8_t Subsampling_420_Colocated = 1;
constexpr uint8_t Subsampling_422 = 2;
constexpr uint8_t Subsampling_444 = 3;
}

namespace VPConfigurationRange {
constexpr uint8_t VideoRange = 0;
constexpr uint8_t FullRange = 1;
}

// Ref: ISO/IEC 23091-2:2019
namespace VPConfigurationColorPrimaries {
constexpr uint8_t BT_709_6 = 1;
constexpr uint8_t Unspecified = 2;
constexpr uint8_t BT_470_6_M = 4;
constexpr uint8_t BT_470_7_BG = 5;
constexpr uint8_t BT_601_7 = 6;
constexpr uint8_t SMPTE_ST_240 = 7;
constexpr uint8_t Film = 8;
constexpr uint8_t BT_2020_Nonconstant_Luminance = 9;
constexpr uint8_t SMPTE_ST_428_1 = 10;
constexpr uint8_t SMPTE_RP_431_2 = 11;
constexpr uint8_t SMPTE_EG_432_1 = 12;
constexpr uint8_t EBU_Tech_3213_E = 22;
}

// Ref: ISO/IEC 23091-2:2019
namespace VPConfigurationTransferCharacteristics {
constexpr uint8_t BT_709_6 = 1;
constexpr uint8_t Unspecified = 2;
constexpr uint8_t BT_470_6_M = 4;
constexpr uint8_t BT_470_7_BG = 5;
constexpr uint8_t BT_601_7 = 6;
constexpr uint8_t SMPTE_ST_240 = 7;
constexpr uint8_t Linear = 8;
constexpr uint8_t Logrithmic = 9;
constexpr uint8_t Logrithmic_Sqrt = 10;
constexpr uint8_t IEC_61966_2_4 = 11;
constexpr uint8_t BT_1361_0 = 12;
constexpr uint8_t IEC_61966_2_1 = 13;
constexpr uint8_t BT_2020_10bit = 14;
constexpr uint8_t BT_2020_12bit = 15;
constexpr uint8_t SMPTE_ST_2084 = 16;
constexpr uint8_t SMPTE_ST_428_1 = 17;
constexpr uint8_t BT_2100_HLG = 18;
}

namespace VPConfigurationMatrixCoefficients {
constexpr uint8_t Identity = 0;
constexpr uint8_t BT_709_6 = 1;
constexpr uint8_t Unspecified = 2;
constexpr uint8_t FCC = 4;
constexpr uint8_t BT_470_7_BG = 5;
constexpr uint8_t BT_601_7 = 6;
constexpr uint8_t SMPTE_ST_240 = 7;
constexpr uint8_t YCgCo = 8;
constexpr uint8_t BT_2020_Nonconstant_Luminance = 9;
constexpr uint8_t BT_2020_Constant_Luminance = 10;
constexpr uint8_t SMPTE_ST_2085 = 11;
constexpr uint8_t Chromacity_Constant_Luminance = 12;
constexpr uint8_t Chromacity_Nonconstant_Luminance = 13;
constexpr uint8_t BT_2100_ICC = 14;
}

struct VPCodecConfigurationRecord {
    String codecName;
    uint8_t profile { 0 };
    uint8_t level { VPConfigurationLevel::Level_1 };
    uint8_t bitDepth { 8 };
    uint8_t chromaSubsampling { VPConfigurationChromaSubsampling::Subsampling_420_Colocated };
    uint8_t videoFullRangeFlag { VPConfigurationRange::VideoRange };
    uint8_t colorPrimaries { VPConfigurationColorPrimaries::BT_709_6 };
    uint8_t transferCharacteristics { VPConfigurationTransferCharacteristics::BT_709_6 };
    uint8_t matrixCoefficients { VPConfigurationMatrixCoefficients::BT_709_6 };
};

WEBCORE_EXPORT std::optional<VPCodecConfigurationRecord> parseVPCodecParameters(StringView codecString);
WEBCORE_EXPORT String createVPCodecParametersString(const VPCodecConfigurationRecord&);
std::optional<VPCodecConfigurationRecord> createVPCodecConfigurationRecordFromVPCC(std::span<const uint8_t>);
void setConfigurationColorSpaceFromVP9ColorSpace(VPCodecConfigurationRecord&, uint8_t);

enum class VPXCodec : uint8_t { Vp8, Vp9 };
std::optional<VPCodecConfigurationRecord> vPCodecConfigurationRecordFromVPXByteStream(VPXCodec, std::span<const uint8_t>);
Vector<uint8_t> vpcCFromVPCodecConfigurationRecord(const VPCodecConfigurationRecord&);

PlatformVideoColorSpace colorSpaceFromVPCodecConfigurationRecord(const VPCodecConfigurationRecord&);

}
