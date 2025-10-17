/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 2, 2024.
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
#include "ImageUtilities.h"

#include "FloatRect.h"
#include "GraphicsContext.h"
#include "ImageDecoderCG.h"
#include "Logging.h"
#include "MIMETypeRegistry.h"
#include "UTIRegistry.h"
#include "UTIUtilities.h"
#include <CoreFoundation/CoreFoundation.h>
#include <ImageIO/ImageIO.h>
#include <WebCore/ShareableBitmap.h>
#include <wtf/FileSystem.h>
#include <wtf/cf/VectorCF.h>
#include <wtf/text/CString.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WorkQueue& sharedImageTranscodingQueue()
{
    static NeverDestroyed<Ref<WorkQueue>> queue(WorkQueue::create("com.apple.WebKit.ImageTranscoding"_s));
    return queue.get();
}

static String transcodeImage(const String& path, const String& destinationUTI, const String& destinationExtension)
{
    auto sourceURL = adoptCF(CFURLCreateWithFileSystemPath(kCFAllocatorDefault, path.createCFString().get(), kCFURLPOSIXPathStyle, false));
    auto source = adoptCF(CGImageSourceCreateWithURL(sourceURL.get(), nullptr));
    if (!source)
        return nullString();

    auto sourceUTI = String(CGImageSourceGetType(source.get()));
    if (!sourceUTI || sourceUTI == destinationUTI)
        return nullString();

    // It is important to add the appropriate file extension to the temporary file path.
    // The File object depends solely on the extension to know the MIME type of the file.
    auto suffix = makeString('.', destinationExtension);
    auto [destinationPath, destinationFileHandle] = FileSystem::openTemporaryFile("tempImage"_s, suffix);
    if (destinationFileHandle == FileSystem::invalidPlatformFileHandle) {
        RELEASE_LOG_ERROR(Images, "transcodeImage: Destination image could not be created: %s %s\n", path.utf8().data(), destinationUTI.utf8().data());
        return nullString();
    }

    CGDataConsumerCallbacks callbacks = {
        [](void* info, const void* buffer, size_t count) -> size_t {
            auto handle = *static_cast<FileSystem::PlatformFileHandle*>(info);
            return FileSystem::writeToFile(handle, unsafeMakeSpan(static_cast<const uint8_t*>(buffer), count));
        },
        nullptr
    };

    auto consumer = adoptCF(CGDataConsumerCreate(&destinationFileHandle, &callbacks));
    auto destination = adoptCF(CGImageDestinationCreateWithDataConsumer(consumer.get(), destinationUTI.createCFString().get(), 1, nullptr));

    CGImageDestinationAddImageFromSource(destination.get(), source.get(), 0, nullptr);

    if (!CGImageDestinationFinalize(destination.get())) {
        RELEASE_LOG_ERROR(Images, "transcodeImage: Image transcoding fails: %s %s\n", path.utf8().data(), destinationUTI.utf8().data());
        FileSystem::closeFile(destinationFileHandle);
        FileSystem::deleteFile(destinationPath);
        return nullString();
    }

    FileSystem::closeFile(destinationFileHandle);
    return destinationPath;
}

Vector<String> findImagesForTranscoding(const Vector<String>& paths, const Vector<String>& allowedMIMETypes)
{
    bool needsTranscoding = false;
    auto transcodingPaths = paths.map([&](auto& path) {
        // Append a path of the image which needs transcoding. Otherwise append a null string.
        if (!allowedMIMETypes.contains(WebCore::MIMETypeRegistry::mimeTypeForPath(path))) {
            needsTranscoding = true;
            return path;
        }
        return nullString();
    });

    // If none of the files needs image transcoding, return an empty Vector.
    return needsTranscoding ? transcodingPaths : Vector<String>();
}

Vector<String> transcodeImages(const Vector<String>& paths, const String& destinationUTI, const String& destinationExtension)
{
    ASSERT(!destinationUTI.isNull());
    ASSERT(!destinationExtension.isNull());
    
    return paths.map([&](auto& path) {
        // Append the transcoded path if the image needs transcoding. Otherwise append a null string.
        return path.isNull() ? nullString() : transcodeImage(path, destinationUTI, destinationExtension);
    });
}

String descriptionString(ImageDecodingError error)
{
    switch (error) {
    case ImageDecodingError::Internal:
        return "Internal error"_s;
    case ImageDecodingError::BadData:
        return "Bad data"_s;
    case ImageDecodingError::UnsupportedType:
        return "Unsupported image type"_s;
    }

    return "Unkown error"_s;
}

Expected<std::pair<String, Vector<IntSize>>, ImageDecodingError> utiAndAvailableSizesFromImageData(std::span<const uint8_t> data)
{
    Ref buffer = FragmentedSharedBuffer::create(data);
    Ref imageDecoder = ImageDecoderCG::create(buffer.get(), AlphaOption::Premultiplied, GammaAndColorProfileOption::Applied);
    imageDecoder->setData(buffer.get(), true);
    if (imageDecoder->encodedDataStatus() == EncodedDataStatus::Error)
        return makeUnexpected(ImageDecodingError::BadData);

    auto uti = imageDecoder->uti();
    if (!isSupportedImageType(uti))
        return makeUnexpected(ImageDecodingError::UnsupportedType);

    size_t frameCount = imageDecoder->frameCount();
    // Do not support animated image.
    if (imageDecoder->repetitionCount() != RepetitionCountNone && frameCount > 1)
        return makeUnexpected(ImageDecodingError::UnsupportedType);

    Vector<IntSize> sizes;
    sizes.reserveInitialCapacity(frameCount);
    for (size_t index = 0; index < frameCount; ++index)
        sizes.append(imageDecoder->frameSizeAtIndex(index));

    return std::make_pair(WTFMove(uti), WTFMove(sizes));
}

static RefPtr<NativeImage> createNativeImageFromData(std::span<const uint8_t> data, std::optional<FloatSize> preferredSize)
{
    Ref buffer = FragmentedSharedBuffer::create(data);
    Ref imageDecoder = ImageDecoderCG::create(buffer.get(), AlphaOption::Premultiplied, GammaAndColorProfileOption::Applied);
    imageDecoder->setData(buffer.get(), true);
    if (imageDecoder->encodedDataStatus() == EncodedDataStatus::Error)
        return nullptr;

    auto sourceUTI = imageDecoder->uti();
    if (!isSupportedImageType(sourceUTI))
        return nullptr;

    auto preferredIndex = [&]() -> size_t {
        if (!preferredSize)
            return imageDecoder->primaryFrameIndex();
        size_t frameCount = imageDecoder->frameCount();
        for (size_t index = 0; index < frameCount; ++index) {
            if (imageDecoder->frameSizeAtIndex(index) == *preferredSize)
                return index;
        }
        return imageDecoder->primaryFrameIndex();
    };
    RetainPtr image = imageDecoder->createFrameImageAtIndex(preferredIndex());
    if (!image)
        return nullptr;

    return NativeImage::create(WTFMove(image));
}

static Vector<Ref<ShareableBitmap>> createBitmapsFromNativeImage(NativeImage& image, std::span<const unsigned> lengths)
{
    Vector<Ref<ShareableBitmap>> bitmaps;
    auto sourceColorSpace = image.colorSpace();
    // The conversion could lead to loss of HDR contents.
    auto destinationColorSpace = sourceColorSpace.supportsOutput() ? sourceColorSpace : DestinationColorSpace::SRGB();
    for (auto length : lengths) {
        RefPtr bitmap = ShareableBitmap::createFromImageDraw(image, destinationColorSpace, { (int)length, (int)length }, image.size());
        if (!bitmap)
            return { };

        bitmaps.append(bitmap.releaseNonNull());
    }

    return bitmaps;
}

void createBitmapsFromImageData(std::span<const uint8_t> data, std::span<const unsigned> lengths, CompletionHandler<void(Vector<Ref<ShareableBitmap>>&&)>&& completionHandler)
{
    if (RefPtr nativeImage = createNativeImageFromData(data, std::nullopt)) {
        completionHandler(createBitmapsFromNativeImage(*nativeImage, lengths));
        return;
    }

    return completionHandler({ });
}

RefPtr<SharedBuffer> createIconDataFromBitmaps(Vector<Ref<ShareableBitmap>>&& bitmaps)
{
    if (bitmaps.isEmpty())
            return nullptr;

        constexpr auto icoUTI = "com.microsoft.ico"_s;
        RetainPtr cfUTI = icoUTI.createCFString();
        RetainPtr colorSpace = adoptCF(CGColorSpaceCreateWithName(kCGColorSpaceSRGB));
        RetainPtr destinationData = adoptCF(CFDataCreateMutable(0, 0));
        RetainPtr destination = adoptCF(CGImageDestinationCreateWithData(destinationData.get(), cfUTI.get(), bitmaps.size(), nullptr));

        for (Ref bitmap : bitmaps) {
            RetainPtr cgImage = bitmap->makeCGImageCopy();
            if (!cgImage) {
                RELEASE_LOG_ERROR(Images, "createIconDataFromBitmaps: Fails to create CGImage with size { %d , %d }", bitmap->size().width(), bitmap->size().height());
                return nullptr;
            }

            CGImageDestinationAddImage(destination.get(), cgImage.get(), nullptr);
        }

        if (!CGImageDestinationFinalize(destination.get()))
            return nullptr;

        return SharedBuffer::create(destinationData.get());
}

void decodeImageWithSize(std::span<const uint8_t> data, std::optional<FloatSize> preferredSize, CompletionHandler<void(RefPtr<ShareableBitmap>&&)>&& completionHandler)
{
    auto nativeImage = createNativeImageFromData(data, preferredSize);
    if (!nativeImage)
        return completionHandler(nullptr);

    auto sourceColorSpace = nativeImage->colorSpace();
    auto destinationColorSpace = sourceColorSpace.supportsOutput() ? sourceColorSpace : DestinationColorSpace::SRGB();
    RefPtr bitmap = ShareableBitmap::create({ nativeImage->size(), destinationColorSpace });
    if (!bitmap)
        return completionHandler(nullptr);

    auto context = bitmap->createGraphicsContext();
    if (!context)
        return completionHandler(nullptr);

    FloatRect rect { { }, nativeImage->size() };
    context->drawNativeImage(*nativeImage, rect, rect, { CompositeOperator::Copy });
    completionHandler(WTFMove(bitmap));
}

} // namespace WebCore
