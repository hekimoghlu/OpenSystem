/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 26, 2022.
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

#if ENABLE(WEB_CODECS)

#include "AvcEncoderConfig.h"
#include "BitrateMode.h"
#include "HardwareAcceleration.h"
#include "LatencyMode.h"
#include "WebCodecsAlphaOption.h"
#include <optional>

namespace WebCore {

struct WebCodecsVideoEncoderConfig {
    String codec;
    size_t width;
    size_t height;
    std::optional<size_t> displayWidth;
    std::optional<size_t> displayHeight;
    std::optional<uint64_t> bitrate;
    std::optional<double> framerate;
    HardwareAcceleration hardwareAcceleration { HardwareAcceleration::NoPreference };
    WebCodecsAlphaOption alpha { WebCodecsAlphaOption::Discard };
    String scalabilityMode;
    BitrateMode bitrateMode { BitrateMode::Variable };
    LatencyMode latencyMode { LatencyMode::Quality };
    std::optional<AvcEncoderConfig> avc;

    WebCodecsVideoEncoderConfig isolatedCopy() && { return { WTFMove(codec).isolatedCopy(), width, height, displayWidth, displayHeight, bitrate, framerate, hardwareAcceleration, alpha, WTFMove(scalabilityMode).isolatedCopy(), bitrateMode, latencyMode, avc }; }
    WebCodecsVideoEncoderConfig isolatedCopy() const & { return { codec.isolatedCopy(), width, height, displayWidth, displayHeight, bitrate, framerate, hardwareAcceleration, alpha, scalabilityMode.isolatedCopy(), bitrateMode, latencyMode, avc }; }
};

inline bool isSameConfigurationExceptBitrateAndFramerate(const WebCodecsVideoEncoderConfig& a, const WebCodecsVideoEncoderConfig& b)
{
    return a.codec == b.codec
        && a.width == b.width
        && a.height == b.height
        && a.displayWidth == b.displayWidth
        && a.displayHeight == b.displayHeight
        && a.hardwareAcceleration == b.hardwareAcceleration
        && a.alpha == b.alpha
        && a.scalabilityMode == b.scalabilityMode
        && a.bitrateMode == b.bitrateMode
        && a.latencyMode == b.latencyMode
        && (!!a.avc == !!b.avc)
        && (!a.avc || (a.avc->format == b.avc->format));
}

}

#endif // ENABLE(WEB_CODECS)
