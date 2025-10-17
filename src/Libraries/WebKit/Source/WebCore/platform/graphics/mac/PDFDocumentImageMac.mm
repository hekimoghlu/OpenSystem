/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 28, 2023.
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
#import "config.h"
#import "PDFDocumentImage.h"

#if USE(PDFKIT_FOR_PDFDOCUMENTIMAGE)

#import "LocalCurrentGraphicsContext.h"
#import "SharedBuffer.h"
#import <Quartz/Quartz.h>
#import <objc/objc-class.h>
#import <pal/spi/cg/CoreGraphicsSPI.h>
#import <wtf/RetainPtr.h>
#import <wtf/SoftLinking.h>

SOFT_LINK_FRAMEWORK_IN_UMBRELLA(Quartz, PDFKit)
SOFT_LINK_CLASS(PDFKit, PDFDocument)

extern "C" {
bool CGContextGetAllowsFontSmoothing(CGContextRef context);
bool CGContextGetAllowsFontSubpixelQuantization(CGContextRef context);
}

namespace WebCore {

void PDFDocumentImage::createPDFDocument()
{
    m_document = adoptNS([allocPDFDocumentInstance() initWithData:data()->makeContiguous()->createNSData().get()]);
}

void PDFDocumentImage::computeBoundsForCurrentPage()
{
    PDFPage *pdfPage = [m_document pageAtIndex:0];

    m_cropBox = [pdfPage boundsForBox:kPDFDisplayBoxCropBox];
    m_rotationDegrees = [pdfPage rotation];
}

unsigned PDFDocumentImage::pageCount() const
{
    return [m_document pageCount];
}

void PDFDocumentImage::drawPDFPage(GraphicsContext& context)
{
    LocalCurrentGraphicsContext localCurrentContext(context);

    // These states can be mutated by PDFKit but are not saved
    // on the context's state stack. (<rdar://problem/14951759&35738181>)
    bool allowsSmoothing = CGContextGetAllowsFontSmoothing(context.platformContext());
    bool allowsSubpixelQuantization = CGContextGetAllowsFontSubpixelQuantization(context.platformContext());
    bool allowsSubpixelPositioning = CGContextGetAllowsFontSubpixelPositioning(context.platformContext());

ALLOW_DEPRECATED_DECLARATIONS_BEGIN
    [[m_document pageAtIndex:0] drawWithBox:kPDFDisplayBoxCropBox];
ALLOW_DEPRECATED_DECLARATIONS_END

    CGContextSetAllowsFontSmoothing(context.platformContext(), allowsSmoothing);
    CGContextSetAllowsFontSubpixelQuantization(context.platformContext(), allowsSubpixelQuantization);
    CGContextSetAllowsFontSubpixelPositioning(context.platformContext(), allowsSubpixelPositioning);
}

}

#endif // USE(PDFKIT_FOR_PDFDOCUMENTIMAGE)
