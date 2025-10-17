/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 1, 2022.
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
#include "AV1UtilitiesCocoa.h"

#if PLATFORM(COCOA) && ENABLE(AV1)

#import "AV1Utilities.h"
#import "MediaCapabilitiesInfo.h"
#import <wtf/cf/TypeCastsCF.h>
#import <wtf/cf/VectorCF.h>
#import <wtf/cocoa/TypeCastsCocoa.h>
#import <wtf/text/StringToIntegerConversion.h>

#import "VideoToolboxSoftLink.h"

namespace WebCore {

static bool isConfigurationRecordHDR(const AV1CodecConfigurationRecord& record)
{
    if (record.bitDepth < 10)
        return false;

    if (record.colorPrimaries != static_cast<uint8_t>(AV1ConfigurationColorPrimaries::BT_2020_Nonconstant_Luminance))
        return false;

    if (record.transferCharacteristics != static_cast<uint8_t>(AV1ConfigurationTransferCharacteristics::BT_2020_10bit)
        && record.transferCharacteristics != static_cast<uint8_t>(AV1ConfigurationTransferCharacteristics::BT_2020_12bit)
        && record.transferCharacteristics != static_cast<uint8_t>(AV1ConfigurationTransferCharacteristics::SMPTE_ST_2084)
        && record.transferCharacteristics != static_cast<uint8_t>(AV1ConfigurationTransferCharacteristics::BT_2100_HLG))
        return false;

    if (record.matrixCoefficients != static_cast<uint8_t>(AV1ConfigurationMatrixCoefficients::BT_2020_Nonconstant_Luminance)
        && record.matrixCoefficients != static_cast<uint8_t>(AV1ConfigurationMatrixCoefficients::BT_2020_Constant_Luminance)
        && record.matrixCoefficients != static_cast<uint8_t>(AV1ConfigurationMatrixCoefficients::BT_2100_ICC))
        return false;

    return true;
}

std::optional<MediaCapabilitiesInfo> validateAV1Parameters(const AV1CodecConfigurationRecord& record, const VideoConfiguration& configuration)
{
    if (!validateAV1ConfigurationRecord(record))
        return std::nullopt;

    if (!validateAV1PerLevelConstraints(record, configuration))
        return std::nullopt;

    if (!canLoad_VideoToolbox_VTCopyAV1DecoderCapabilitiesDictionary()
        || !canLoad_VideoToolbox_kVTDecoderCodecCapability_SupportedProfiles()
        || !canLoad_VideoToolbox_kVTDecoderCodecCapability_PerProfileSupport()
        || !canLoad_VideoToolbox_kVTDecoderProfileCapability_IsHardwareAccelerated()
        || !canLoad_VideoToolbox_kVTDecoderProfileCapability_MaxDecodeLevel()
        || !canLoad_VideoToolbox_kVTDecoderProfileCapability_MaxPlaybackLevel()
        || !canLoad_VideoToolbox_kVTDecoderCapability_ChromaSubsampling()
        || !canLoad_VideoToolbox_kVTDecoderCapability_ColorDepth())
        return std::nullopt;

    auto capabilities = adoptCF(softLink_VideoToolbox_VTCopyAV1DecoderCapabilitiesDictionary());
    if (!capabilities)
        return std::nullopt;

    auto supportedProfiles = dynamic_cf_cast<CFArrayRef>(CFDictionaryGetValue(capabilities.get(), kVTDecoderCodecCapability_SupportedProfiles));
    if (!supportedProfiles)
        return std::nullopt;

    int16_t profile = static_cast<int16_t>(record.profile);
    auto cfProfile = adoptCF(CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt16Type, &profile));
    auto searchRange = CFRangeMake(0, CFArrayGetCount(supportedProfiles));
    if (!CFArrayContainsValue(supportedProfiles, searchRange, cfProfile.get()))
        return std::nullopt;

    auto perProfileSupport = dynamic_cf_cast<CFDictionaryRef>(CFDictionaryGetValue(capabilities.get(), kVTDecoderCodecCapability_PerProfileSupport));
    if (!perProfileSupport)
        return std::nullopt;

    auto profileString = String::number(profile).createCFString();
    auto profileSupport = dynamic_cf_cast<CFDictionaryRef>(CFDictionaryGetValue(perProfileSupport, profileString.get()));
    if (!profileSupport)
        return std::nullopt;

    MediaCapabilitiesInfo info;

    info.supported = true;

    info.powerEfficient = CFDictionaryGetValue(profileSupport, kVTDecoderProfileCapability_IsHardwareAccelerated) == kCFBooleanTrue;

    if (auto cfMaxDecodeLevel = dynamic_cf_cast<CFNumberRef>(CFDictionaryGetValue(profileSupport, kVTDecoderProfileCapability_MaxDecodeLevel))) {
        int16_t maxDecodeLevel = 0;
        if (!CFNumberGetValue(cfMaxDecodeLevel, kCFNumberSInt16Type, &maxDecodeLevel))
            return std::nullopt;

        if (static_cast<int16_t>(record.level) > maxDecodeLevel)
            return std::nullopt;
    }

    if (auto cfSupportedChromaSubsampling = dynamic_cf_cast<CFArrayRef>(CFDictionaryGetValue(profileSupport, kVTDecoderCapability_ChromaSubsampling))) {
        auto supportedChromaSubsampling = makeVector(cfSupportedChromaSubsampling, [](CFStringRef chromaSubsamplingString) {
            return parseInteger<uint8_t>(String(chromaSubsamplingString));
        });

        // CoreMedia defines the kVTDecoderCapability_ChromaSubsampling value as
        // three decimal digits consisting of, in order from highest digit to lowest:
        // [subsampling_x, subsampling_y, mono_chrome]. This conflicts with AV1's
        // definition of chromaSubsampling in the Codecs Parameter String:
        // "The chromaSubsampling parameter value, represented by a three-digit decimal,
        // SHALL have its first digit equal to subsampling_x and its second digit equal to
        // subsampling_y. If both subsampling_x and subsampling_y are set to 1, then the third
        // digit SHALL be equal to chroma_sample_position, otherwise it SHALL be set to 0."

        // CoreMedia supports all values of chroma_sample_position, so to reconcile this
        // discrepency, construct a "chroma subsampling" query out of the high-order digits
        // of the AV1CodecConfigurationRecord.chromaSubsampling field, and use the
        // AV1CodecConfigurationRecord.monochrome field as the low-order digit.

        uint8_t subsamplingXandY = record.chromaSubsampling - (record.chromaSubsampling % 10);
        uint8_t subsamplingQuery = subsamplingXandY + (record.monochrome ? 1 : 0);

        if (!supportedChromaSubsampling.contains(subsamplingQuery))
            return std::nullopt;
    }

    if (auto cfSupportedColorDepths = dynamic_cf_cast<CFArrayRef>(CFDictionaryGetValue(profileSupport, kVTDecoderCapability_ColorDepth))) {
        auto supportedColorDepths = makeVector(cfSupportedColorDepths, [](CFStringRef colorDepthString) {
            return parseInteger<uint8_t>(String(colorDepthString));
        });
        if (!supportedColorDepths.contains(static_cast<uint8_t>(record.bitDepth)))
            return std::nullopt;
    }

    if (auto cfMaxPlaybackLevel = dynamic_cf_cast<CFNumberRef>(CFDictionaryGetValue(profileSupport, kVTDecoderProfileCapability_MaxPlaybackLevel))) {
        int16_t maxPlaybackLevel = 0;
        if (!CFNumberGetValue(cfMaxPlaybackLevel, kCFNumberSInt16Type, &maxPlaybackLevel))
            return std::nullopt;

        info.smooth = static_cast<int16_t>(record.level) <= maxPlaybackLevel;
    }

    if (canLoad_VideoToolbox_kVTDecoderProfileCapability_MaxHDRPlaybackLevel() && isConfigurationRecordHDR(record)) {
        if (auto cfMaxHDRPlaybackLevel = dynamic_cf_cast<CFNumberRef>(CFDictionaryGetValue(profileSupport, kVTDecoderProfileCapability_MaxHDRPlaybackLevel))) {
            int16_t maxHDRPlaybackLevel = 0;
            if (!CFNumberGetValue(cfMaxHDRPlaybackLevel, kCFNumberSInt16Type, &maxHDRPlaybackLevel))
                return std::nullopt;

            info.smooth = static_cast<int16_t>(record.level) <= maxHDRPlaybackLevel;
        }
    }

    return info;
}

bool av1HardwareDecoderAvailable()
{
    if (canLoad_VideoToolbox_VTIsHardwareDecodeSupported())
        return VTIsHardwareDecodeSupported('av01');

    return false;
}

}

#endif
