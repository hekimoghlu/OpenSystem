/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 1, 2024.
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
#include "MediaCapabilitiesLogging.h"

#include "AudioConfiguration.h"
#include "JSColorGamut.h"
#include "JSHdrMetadataType.h"
#include "JSMediaDecodingType.h"
#include "JSMediaEncodingType.h"
#include "JSTransferFunction.h"
#include "MediaCapabilitiesDecodingInfo.h"
#include "MediaCapabilitiesEncodingInfo.h"
#include "MediaDecodingConfiguration.h"
#include "MediaDecodingType.h"
#include "MediaEncodingConfiguration.h"
#include "MediaEncodingType.h"
#include "VideoConfiguration.h"
#include <wtf/JSONValues.h>

namespace WebCore {

static Ref<JSON::Object> toJSONObject(const VideoConfiguration& configuration)
{
    auto object = JSON::Object::create();
    object->setString("contentType"_s, configuration.contentType);
    object->setInteger("width"_s, configuration.width);
    object->setInteger("height"_s, configuration.height);
    object->setInteger("bitrate"_s, static_cast<int>(configuration.bitrate));
    object->setDouble("framerate"_s, configuration.framerate);
    if (configuration.alphaChannel)
        object->setBoolean("alphaChannel"_s, configuration.alphaChannel.value());
    if (configuration.colorGamut)
        object->setString("colorGamut"_s, convertEnumerationToString(configuration.colorGamut.value()));
    if (configuration.hdrMetadataType)
        object->setString("hdrMetadataType"_s, convertEnumerationToString(configuration.hdrMetadataType.value()));
    if (configuration.transferFunction)
        object->setString("transferFunction"_s, convertEnumerationToString(configuration.transferFunction.value()));
    return object;
}

static Ref<JSON::Object> toJSONObject(const AudioConfiguration& configuration)
{
    auto object = JSON::Object::create();
    object->setString("contentType"_s, configuration.contentType);
    if (!configuration.channels.isNull())
        object->setString("channels"_s, configuration.channels);
    if (configuration.bitrate)
        object->setInteger("bitrate"_s, static_cast<int>(configuration.bitrate.value()));
    if (configuration.samplerate)
        object->setDouble("samplerate"_s, configuration.samplerate.value());
    if (configuration.spatialRendering)
        object->setBoolean("spatialRendering"_s, configuration.spatialRendering.value());
    return object;
}

static Ref<JSON::Object> toJSONObject(const MediaConfiguration& configuration)
{
    auto object = JSON::Object::create();
    if (configuration.video)
        object->setValue("video"_s, toJSONObject(configuration.video.value()));
    if (configuration.audio)
        object->setValue("audio"_s, toJSONObject(configuration.audio.value()));
    return object;
}

static Ref<JSON::Object> toJSONObject(const MediaDecodingConfiguration& configuration)
{
    auto object = toJSONObject(static_cast<const MediaConfiguration&>(configuration));
    object->setString("type"_s, convertEnumerationToString(configuration.type));
    return object;
}

static Ref<JSON::Object> toJSONObject(const MediaEncodingConfiguration& configuration)
{
    auto object = toJSONObject(static_cast<const MediaConfiguration&>(configuration));
    object->setString("type"_s, convertEnumerationToString(configuration.type));
    return object;
}

static Ref<JSON::Object> toJSONObject(const MediaCapabilitiesInfo& info)
{
    auto object = JSON::Object::create();
    object->setBoolean("supported"_s, info.supported);
    object->setBoolean("smooth"_s, info.smooth);
    object->setBoolean("powerEfficient"_s, info.powerEfficient);
    return object;
}

static Ref<JSON::Object> toJSONObject(const MediaCapabilitiesDecodingInfo& info)
{
    auto object = toJSONObject(static_cast<const MediaCapabilitiesInfo&>(info));
    object->setValue("supportedConfiguration"_s, toJSONObject(info.supportedConfiguration));
    return object;
}

static Ref<JSON::Object> toJSONObject(const MediaCapabilitiesEncodingInfo& info)
{
    auto object = toJSONObject(static_cast<const MediaCapabilitiesInfo&>(info));
    object->setValue("supportedConfiguration"_s, toJSONObject(info.supportedConfiguration));
    return object;
}

static String toJSONString(const VideoConfiguration& configuration)
{
    return toJSONObject(configuration)->toJSONString();
}

static String toJSONString(const AudioConfiguration& configuration)
{
    return toJSONObject(configuration)->toJSONString();
}

static String toJSONString(const MediaConfiguration& configuration)
{
    return toJSONObject(configuration)->toJSONString();
}

static String toJSONString(const MediaDecodingConfiguration& configuration)
{
    return toJSONObject(configuration)->toJSONString();
}

static String toJSONString(const MediaEncodingConfiguration& configuration)
{
    return toJSONObject(configuration)->toJSONString();
}

static String toJSONString(const MediaCapabilitiesInfo& info)
{
    return toJSONObject(info)->toJSONString();
}

static String toJSONString(const MediaCapabilitiesDecodingInfo& info)
{
    return toJSONObject(info)->toJSONString();
}

static String toJSONString(const MediaCapabilitiesEncodingInfo& info)
{
    return toJSONObject(info)->toJSONString();
}

}

namespace WTF {

String LogArgument<WebCore::VideoConfiguration>::toString(const WebCore::VideoConfiguration& configuration)
{
    return toJSONString(configuration);
}

String LogArgument<WebCore::AudioConfiguration>::toString(const WebCore::AudioConfiguration& configuration)
{
    return toJSONString(configuration);
}

String LogArgument<WebCore::MediaConfiguration>::toString(const WebCore::MediaConfiguration& configuration)
{
    return toJSONString(configuration);
}

String LogArgument<WebCore::MediaDecodingConfiguration>::toString(const WebCore::MediaDecodingConfiguration& configuration)
{
    return toJSONString(configuration);
}

String LogArgument<WebCore::MediaEncodingConfiguration>::toString(const WebCore::MediaEncodingConfiguration& configuration)
{
    return toJSONString(configuration);
}

String LogArgument<WebCore::MediaCapabilitiesInfo>::toString(const WebCore::MediaCapabilitiesInfo& info)
{
    return toJSONString(info);
}

String LogArgument<WebCore::MediaCapabilitiesDecodingInfo>::toString(const WebCore::MediaCapabilitiesDecodingInfo& info)
{
    return toJSONString(info);
}

String LogArgument<WebCore::MediaCapabilitiesEncodingInfo>::toString(const WebCore::MediaCapabilitiesEncodingInfo& info)
{
    return toJSONString(info);
}

String LogArgument<WebCore::ColorGamut>::toString(const WebCore::ColorGamut& type)
{
    return convertEnumerationToString(type);
}

String LogArgument<WebCore::HdrMetadataType>::toString(const WebCore::HdrMetadataType& type)
{
    return convertEnumerationToString(type);
}

String LogArgument<WebCore::TransferFunction>::toString(const WebCore::TransferFunction& type)
{
    return convertEnumerationToString(type);
}

String LogArgument<WebCore::MediaDecodingType>::toString(const WebCore::MediaDecodingType& type)
{
    return convertEnumerationToString(type);
}

String LogArgument<WebCore::MediaEncodingType>::toString(const WebCore::MediaEncodingType& type)
{
    return convertEnumerationToString(type);
}

}
