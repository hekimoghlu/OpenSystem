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
#include "ImageDecoder.h"

#include "ImageFrame.h"
#include "ScalableImageDecoder.h"
#include <wtf/NeverDestroyed.h>
#include <wtf/TZoneMallocInlines.h>

#if USE(CG)
#include "ImageDecoderCG.h"
#endif

#if HAVE(AVASSETREADER)
#include "ImageDecoderAVFObjC.h"
#endif

#if USE(GSTREAMER) && ENABLE(VIDEO)
#include "ImageDecoderGStreamer.h"
#endif

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(ImageDecoder);

#if ENABLE(GPU_PROCESS) && HAVE(AVASSETREADER)
using FactoryVector = Vector<ImageDecoder::ImageDecoderFactory>;

static RefPtr<ImageDecoder> createInProcessImageDecoderAVFObjC(FragmentedSharedBuffer& buffer, const String& mimeType, AlphaOption alphaOption, GammaAndColorProfileOption gammaOption)
{
    return ImageDecoderAVFObjC::create(buffer, mimeType, alphaOption, gammaOption, ProcessIdentity { ProcessIdentity::CurrentProcess });
}

static void platformRegisterFactories(FactoryVector& factories)
{
    factories.append({ ImageDecoderAVFObjC::supportsMediaType, ImageDecoderAVFObjC::canDecodeType, createInProcessImageDecoderAVFObjC });
}

static FactoryVector& installedFactories()
{
    static NeverDestroyed<FactoryVector> factories;
    static std::once_flag registerDefaults;
    std::call_once(registerDefaults, [&] {
        platformRegisterFactories(factories);
    });

    return factories;
}

void ImageDecoder::installFactory(ImageDecoder::ImageDecoderFactory&& factory)
{
    installedFactories().append(WTFMove(factory));
}

void ImageDecoder::resetFactories()
{
    installedFactories().clear();
    platformRegisterFactories(installedFactories());
}

void ImageDecoder::clearFactories()
{
    installedFactories().clear();
}
#endif

RefPtr<ImageDecoder> ImageDecoder::create(FragmentedSharedBuffer& data, const String& mimeType, AlphaOption alphaOption, GammaAndColorProfileOption gammaAndColorProfileOption)
{
    UNUSED_PARAM(mimeType);

#if HAVE(AVASSETREADER)
    if (!ImageDecoderCG::canDecodeType(mimeType)) {
#if ENABLE(GPU_PROCESS)
        for (auto& factory : installedFactories()) {
            if (factory.canDecodeType(mimeType))
                return factory.createImageDecoder(data, mimeType, alphaOption, gammaAndColorProfileOption);
        }
#else
        if (ImageDecoderAVFObjC::canDecodeType(mimeType))
            return ImageDecoderAVFObjC::create(data, mimeType, alphaOption, gammaAndColorProfileOption);
#endif
    }
#endif

#if USE(GSTREAMER) && ENABLE(VIDEO)
    if (ImageDecoderGStreamer::canDecodeType(mimeType))
        return ImageDecoderGStreamer::create(data, mimeType, alphaOption, gammaAndColorProfileOption);
#endif

#if USE(CG)
    // ScalableImageDecoder is used on CG ports for some specific image formats which the platform doesn't support directly.
    if (auto imageDecoder = ScalableImageDecoder::create(data, alphaOption, gammaAndColorProfileOption))
        return imageDecoder;
    return ImageDecoderCG::create(data, alphaOption, gammaAndColorProfileOption);
#else
    return ScalableImageDecoder::create(data, alphaOption, gammaAndColorProfileOption);
#endif
}

ImageDecoder::ImageDecoder() = default;

ImageDecoder::~ImageDecoder() = default;

bool ImageDecoder::supportsMediaType(MediaType type)
{
#if USE(CG)
    if (ImageDecoderCG::supportsMediaType(type))
        return true;
#else
    if (ScalableImageDecoder::supportsMediaType(type))
        return true;
#endif

#if HAVE(AVASSETREADER)
#if ENABLE(GPU_PROCESS)
    for (auto& factory : installedFactories()) {
        if (factory.supportsMediaType(type))
            return true;
    }
#else
    if (ImageDecoderAVFObjC::supportsMediaType(type))
        return true;
#endif
#endif

#if USE(GSTREAMER) && ENABLE(VIDEO)
    if (ImageDecoderGStreamer::supportsMediaType(type))
        return true;
#endif

    return false;
}

bool ImageDecoder::fetchFrameMetaDataAtIndex(size_t index, SubsamplingLevel subsamplingLevel, const DecodingOptions& options, ImageFrame& frame) const
{
    if (options.hasSizeForDrawing()) {
        ASSERT(frame.hasNativeImage());
        frame.m_size = frame.nativeImage()->size();
    } else
        frame.m_size = frameSizeAtIndex(index, subsamplingLevel);

    frame.m_densityCorrectedSize = frameDensityCorrectedSizeAtIndex(index);
    frame.m_subsamplingLevel = subsamplingLevel;
    frame.m_decodingOptions = options;
    frame.m_hasAlpha = frameHasAlphaAtIndex(index);
    frame.m_orientation = frameOrientationAtIndex(index);
    frame.m_decodingStatus = frameIsCompleteAtIndex(index) ? DecodingStatus::Complete : DecodingStatus::Partial;
    return true;
}

} // namespace WebCore
