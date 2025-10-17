/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 22, 2023.
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

#include "GStreamerCommon.h"
#include "VideoEncoderScalabilityMode.h"
#include <wtf/TZoneMalloc.h>

#define WEBKIT_TYPE_VIDEO_ENCODER (webkit_video_encoder_get_type())
#define WEBKIT_VIDEO_ENCODER(obj) (G_TYPE_CHECK_INSTANCE_CAST((obj), WEBKIT_TYPE_VIDEO_ENCODER, WebKitVideoEncoder))
#define WEBKIT_VIDEO_ENCODER_CLASS(klass) (G_TYPE_CHECK_CLASS_CAST((klass), WEBKIT_TYPE_VIDEO_ENCODER, WebKitVideoEncoderClass))
#define WEBKIT_IS_VIDEO_ENCODER(obj) (G_TYPE_CHECK_INSTANCE_TYPE((obj), WEBKIT_TYPE_VIDEO_ENCODER))
#define WEBKIT_IS_VIDEO_ENCODER_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass), WEBKIT_TYPE_VIDEO_ENCODER))

typedef struct _WebKitVideoEncoder WebKitVideoEncoder;
typedef struct _WebKitVideoEncoderClass WebKitVideoEncoderClass;
typedef struct _WebKitVideoEncoderPrivate WebKitVideoEncoderPrivate;

struct _WebKitVideoEncoder {
    GstBin parent;

    WebKitVideoEncoderPrivate* priv;
};

struct _WebKitVideoEncoderClass {
    GstBinClass parentClass;
};

GType webkit_video_encoder_get_type(void);

class WebKitVideoEncoderBitRateAllocation : public RefCounted<WebKitVideoEncoderBitRateAllocation> {
    WTF_MAKE_TZONE_ALLOCATED(WebKitVideoEncoderBitRateAllocation);
    WTF_MAKE_NONCOPYABLE(WebKitVideoEncoderBitRateAllocation);

public:
    static Ref<WebKitVideoEncoderBitRateAllocation> create(WebCore::VideoEncoderScalabilityMode scalabilityMode)
    {
        return adoptRef(*new WebKitVideoEncoderBitRateAllocation(scalabilityMode));
    }

    static const unsigned MaxSpatialLayers = 5;
    static const unsigned MaxTemporalLayers = 4;

    void setBitRate(unsigned spatialLayerIndex, unsigned temporalLayerIndex, uint32_t bitRate)
    {
        RELEASE_ASSERT(spatialLayerIndex < MaxSpatialLayers);
        RELEASE_ASSERT(temporalLayerIndex < MaxTemporalLayers);
        m_bitRates[spatialLayerIndex][temporalLayerIndex].emplace(bitRate);
    }

    std::optional<uint32_t> getBitRate(unsigned spatialLayerIndex, unsigned temporalLayerIndex) const
    {
        if (UNLIKELY(spatialLayerIndex >= MaxSpatialLayers))
            return std::nullopt;
        if (UNLIKELY(temporalLayerIndex >= MaxTemporalLayers))
            return std::nullopt;
        return m_bitRates[spatialLayerIndex][temporalLayerIndex];
    }

    WebCore::VideoEncoderScalabilityMode scalabilityMode() const { return m_scalabilityMode; }

private:
    WebKitVideoEncoderBitRateAllocation(WebCore::VideoEncoderScalabilityMode scalabilityMode)
        : m_scalabilityMode(scalabilityMode)
    { }

    WebCore::VideoEncoderScalabilityMode m_scalabilityMode;
    std::array<std::array<std::optional<uint32_t>, MaxSpatialLayers>, MaxTemporalLayers> m_bitRates;
};

bool videoEncoderSupportsCodec(WebKitVideoEncoder*, const String&);
bool videoEncoderSetCodec(WebKitVideoEncoder*, const String&, std::optional<WebCore::IntSize> = std::nullopt, std::optional<double> frameRate = std::nullopt);
void videoEncoderSetBitRateAllocation(WebKitVideoEncoder*, RefPtr<WebKitVideoEncoderBitRateAllocation>&&);
void teardownVideoEncoderSingleton();
