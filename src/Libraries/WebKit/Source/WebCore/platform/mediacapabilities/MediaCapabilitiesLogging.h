/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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

#include <wtf/text/WTFString.h>

namespace WebCore {

struct VideoConfiguration;
struct AudioConfiguration;
struct MediaConfiguration;
struct MediaDecodingConfiguration;
struct MediaEncodingConfiguration;
struct MediaCapabilitiesInfo;
struct MediaCapabilitiesDecodingInfo;
struct MediaCapabilitiesEncodingInfo;

enum class ColorGamut : uint8_t;
enum class HdrMetadataType : uint8_t;
enum class TransferFunction : uint8_t;
enum class MediaDecodingType : uint8_t;
enum class MediaEncodingType : bool;

}

namespace WTF {

template<typename>
struct LogArgument;

template <>
struct LogArgument<WebCore::VideoConfiguration> {
    static String toString(const WebCore::VideoConfiguration&);
};

template <>
struct LogArgument<WebCore::AudioConfiguration> {
    static String toString(const WebCore::AudioConfiguration&);
};

template <>
struct LogArgument<WebCore::MediaConfiguration> {
    static String toString(const WebCore::MediaConfiguration&);
};

template <>
struct LogArgument<WebCore::MediaDecodingConfiguration> {
    static String toString(const WebCore::MediaDecodingConfiguration&);
};

template <>
struct LogArgument<WebCore::MediaEncodingConfiguration> {
    static String toString(const WebCore::MediaEncodingConfiguration&);
};

template <>
struct LogArgument<WebCore::MediaCapabilitiesInfo> {
    static String toString(const WebCore::MediaCapabilitiesInfo&);
};

template <>
struct LogArgument<WebCore::MediaCapabilitiesDecodingInfo> {
    static String toString(const WebCore::MediaCapabilitiesDecodingInfo&);
};

template <>
struct LogArgument<WebCore::MediaCapabilitiesEncodingInfo> {
    static String toString(const WebCore::MediaCapabilitiesEncodingInfo&);
};

template <>
struct LogArgument<WebCore::ColorGamut> {
    static String toString(const WebCore::ColorGamut&);
};

template <>
struct LogArgument<WebCore::HdrMetadataType> {
    static String toString(const WebCore::HdrMetadataType&);
};

template <>
struct LogArgument<WebCore::TransferFunction> {
    static String toString(const WebCore::TransferFunction&);
};

template <>
struct LogArgument<WebCore::MediaDecodingType> {
    static String toString(const WebCore::MediaDecodingType&);
};

template <>
struct LogArgument<WebCore::MediaEncodingType> {
    static String toString(const WebCore::MediaEncodingType&);
};

}
