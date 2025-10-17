/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 9, 2025.
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
#include "PDFDocumentImage.h"

#if USE(CG)

#include "GeometryUtilities.h"
#include "GraphicsContext.h"
#include "ImageBuffer.h"
#include "ImageObserver.h"
#include <CoreGraphics/CGContext.h>
#include <CoreGraphics/CGPDFDocument.h>
#include <pal/spi/cg/CoreGraphicsSPI.h>
#include <wtf/RetainPtr.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

PDFDocumentImage::PDFDocumentImage(ImageObserver* observer)
    : Image(observer)
{
}

PDFDocumentImage::~PDFDocumentImage() = default;

String PDFDocumentImage::filenameExtension() const
{
    return "pdf"_s;
}

FloatSize PDFDocumentImage::size(ImageOrientation) const
{
    FloatSize expandedCropBoxSize = FloatSize(expandedIntSize(m_cropBox.size()));

    if (m_rotationDegrees == 90 || m_rotationDegrees == 270)
        return expandedCropBoxSize.transposedSize();
    return expandedCropBoxSize;
}

void PDFDocumentImage::computeIntrinsicDimensions(Length& intrinsicWidth, Length& intrinsicHeight, FloatSize& intrinsicRatio)
{
    // FIXME: If we want size negotiation with PDF documents as-image, this is the place to implement it (https://bugs.webkit.org/show_bug.cgi?id=12095).
    Image::computeIntrinsicDimensions(intrinsicWidth, intrinsicHeight, intrinsicRatio);
    intrinsicRatio = FloatSize();
}

EncodedDataStatus PDFDocumentImage::dataChanged(bool allDataReceived)
{
    ASSERT(!m_document);
    if (allDataReceived && !m_document) {
        createPDFDocument();

        if (pageCount()) {
            m_hasPage = true;
            computeBoundsForCurrentPage();
        }
    }
    return m_document ? EncodedDataStatus::Complete : EncodedDataStatus::Unknown;
}

void PDFDocumentImage::decodedSizeChanged(size_t newCachedBytes)
{
    if (!m_cachedBytes && !newCachedBytes)
        return;

    if (auto observer = imageObserver())
        observer->decodedSizeChanged(*this, -static_cast<long long>(m_cachedBytes) + newCachedBytes);

    m_cachedBytes = newCachedBytes;
}

bool PDFDocumentImage::shouldDrawFromCachedSubimage(GraphicsContext& context) const
{
    if (mustDrawFromCachedSubimage(context))
        return true;

    auto scaledImageSize = size() * context.scaleFactor();
    return scaledImageSize.area() > CachedSubimage::maxArea;
}

bool PDFDocumentImage::mustDrawFromCachedSubimage(GraphicsContext& context) const
{
    return !context.hasPlatformContext();
}

std::unique_ptr<CachedSubimage> PDFDocumentImage::createCachedSubimage(GraphicsContext& context, const FloatRect& destinationRect, const FloatRect& sourceRect, ImagePaintingOptions options)
{
    ASSERT(shouldDrawFromCachedSubimage(context));

    if (auto cachedSubimage = CachedSubimage::create(context, size(), destinationRect, sourceRect)) {
        auto result = drawPDFDocument(cachedSubimage->imageBuffer().context(), cachedSubimage->destinationRect(), cachedSubimage->sourceRect(), options);
        if (result != ImageDrawResult::DidDraw)
            return nullptr;
        return cachedSubimage;
    }

    if (!mustDrawFromCachedSubimage(context))
        return nullptr;

    if (auto cachedSubimage = CachedSubimage::createPixelated(context, destinationRect, sourceRect)) {
        auto result = drawPDFDocument(cachedSubimage->imageBuffer().context(), cachedSubimage->destinationRect(), cachedSubimage->sourceRect(), options);
        if (result != ImageDrawResult::DidDraw)
            return nullptr;
        return cachedSubimage;
    }

    return nullptr;
}

ImageDrawResult PDFDocumentImage::drawPDFDocument(GraphicsContext& context, const FloatRect& destinationRect, const FloatRect& sourceRect, ImagePaintingOptions options)
{
    ASSERT(context.hasPlatformContext());

    if (!m_document || !m_hasPage)
        return ImageDrawResult::DidNothing;

    GraphicsContextStateSaver stateSaver(context);
    context.setCompositeOperation(options.compositeOperator());

    context.clip(destinationRect);

    // Transform context such that sourceRect is mapped to destinationRect.
    auto scaleFactor = destinationRect.size() / sourceRect.size();
    auto scaledSourceRect = sourceRect;
    scaledSourceRect.scale(scaleFactor);
    context.translate(destinationRect.location() - scaledSourceRect.location());
    context.scale(scaleFactor);

    // Flip the coords.
    context.translate(0, size().height());
    context.scale({ 1, -1 });
    drawPDFPage(context);

    if (auto observer = imageObserver())
        observer->didDraw(*this);

    return ImageDrawResult::DidDraw;
}

ImageDrawResult PDFDocumentImage::drawFromCachedSubimage(GraphicsContext& context, const FloatRect& destinationRect, const FloatRect& sourceRect, ImagePaintingOptions options)
{
    if (!shouldDrawFromCachedSubimage(context))
        return ImageDrawResult::DidNothing;

    auto clippedDestinationRect = intersection(context.clipBounds(), destinationRect);
    if (clippedDestinationRect.isEmpty())
        return ImageDrawResult::DidDraw;

    auto clippedSourceRect = mapRect(clippedDestinationRect, destinationRect, sourceRect);

    // Reset the currect CachedSubimage if it can't be reused for the current drawing.
    if (m_cachedSubimage && !m_cachedSubimage->canBeUsed(context, clippedDestinationRect, clippedSourceRect))
        m_cachedSubimage = nullptr;

    if (!m_cachedSubimage) {
        m_cachedSubimage = createCachedSubimage(context, clippedDestinationRect, clippedSourceRect, options);
        if (m_cachedSubimage)
            ++m_cachedSubimageCreateCountForTesting;
    }

    if (!m_cachedSubimage)
        return ImageDrawResult::DidNothing;

    m_cachedSubimage->draw(context, clippedDestinationRect, clippedSourceRect);
    ++m_cachedSubimageDrawCountForTesting;
    return ImageDrawResult::DidDraw;
}

ImageDrawResult PDFDocumentImage::draw(GraphicsContext& context, const FloatRect& destination, const FloatRect& source, ImagePaintingOptions options)
{
    auto result = drawFromCachedSubimage(context, destination, source, options);
    if (result != ImageDrawResult::DidNothing)
        return result;

    if (mustDrawFromCachedSubimage(context)) {
        LOG_ERROR("ERROR ImageDrawResult GraphicsContext::drawImage() cached subimage could not been drawn");
        return ImageDrawResult::DidNothing;
    }

    return drawPDFDocument(context, destination, source, options);
}

void PDFDocumentImage::destroyDecodedData(bool)
{
    m_cachedSubimage = nullptr;
    decodedSizeChanged(0);
}

#if !USE(PDFKIT_FOR_PDFDOCUMENTIMAGE)

void PDFDocumentImage::createPDFDocument()
{
    auto dataProvider = adoptCF(CGDataProviderCreateWithCFData(data()->makeContiguous()->createCFData().get()));
    m_document = adoptCF(CGPDFDocumentCreateWithProvider(dataProvider.get()));
}

void PDFDocumentImage::computeBoundsForCurrentPage()
{
    ASSERT(pageCount() > 0);
    CGPDFPageRef cgPage = CGPDFDocumentGetPage(m_document.get(), 1);
    CGRect mediaBox = CGPDFPageGetBoxRect(cgPage, kCGPDFMediaBox);

    // Get crop box (not always there). If not, use media box.
    CGRect r = CGPDFPageGetBoxRect(cgPage, kCGPDFCropBox);
    if (!CGRectIsEmpty(r))
        m_cropBox = r;
    else
        m_cropBox = mediaBox;

    m_rotationDegrees = CGPDFPageGetRotationAngle(cgPage);
}

unsigned PDFDocumentImage::pageCount() const
{
    return CGPDFDocumentGetNumberOfPages(m_document.get());
}

static void applyRotationForPainting(GraphicsContext& context, FloatSize size, int rotationDegrees)
{
    if (rotationDegrees == 90)
        context.translate(0, size.height());
    else if (rotationDegrees == 180)
        context.translate(size);
    else if (rotationDegrees == 270)
        context.translate(size.width(), 0);

    context.rotate(-deg2rad(static_cast<float>(rotationDegrees)));
}

void PDFDocumentImage::drawPDFPage(GraphicsContext& context)
{
    applyRotationForPainting(context, size(), m_rotationDegrees);

    context.translate(-m_cropBox.location());

    // CGPDF pages are indexed from 1.
    CGContextDrawPDFPageWithAnnotations(context.platformContext(), CGPDFDocumentGetPage(m_document.get(), 1), nullptr);
}

#endif // !USE(PDFKIT_FOR_PDFDOCUMENTIMAGE)

void PDFDocumentImage::dump(TextStream& ts) const
{
    Image::dump(ts);
    ts.dumpProperty("page-count", pageCount());
    ts.dumpProperty("crop-box", m_cropBox);
    if (m_rotationDegrees)
        ts.dumpProperty("rotation", m_rotationDegrees);
}

}

#endif // USE(CG)
