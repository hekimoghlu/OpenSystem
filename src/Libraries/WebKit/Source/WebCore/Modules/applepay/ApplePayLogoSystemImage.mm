/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 2, 2024.
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
#import "ApplePayLogoSystemImage.h"

#if ENABLE(APPLE_PAY)

#import "FloatRect.h"
#import "GeometryUtilities.h"
#import "GraphicsContext.h"
#import <wtf/NeverDestroyed.h>
#import <wtf/RetainPtr.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebCore {

static NSBundle *passKitBundle()
{
    static NSBundle *passKitBundle;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        passKitBundle = [NSBundle bundleWithURL:[NSURL fileURLWithPath:[FileSystem::systemDirectoryPath() stringByAppendingPathComponent:@"Library/Frameworks/PassKit.framework"] isDirectory:YES]];
    });
    return passKitBundle;
}

static RetainPtr<CGPDFPageRef> loadPassKitPDFPage(NSString *imageName)
{
    NSURL *url = [passKitBundle() URLForResource:imageName withExtension:@"pdf"];
    if (!url)
        return nullptr;
    auto document = adoptCF(CGPDFDocumentCreateWithURL((CFURLRef)url));
    if (!document)
        return nullptr;
    if (!CGPDFDocumentGetNumberOfPages(document.get()))
        return nullptr;
    return CGPDFDocumentGetPage(document.get(), 1);
}

static RetainPtr<CGPDFPageRef> applePayLogoWhite()
{
    static NeverDestroyed<RetainPtr<CGPDFPageRef>> logoPage;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        logoPage.get() = loadPassKitPDFPage(@"PayButtonLogoWhite");
    });
    return logoPage;
}

static RetainPtr<CGPDFPageRef> applePayLogoBlack()
{
    static NeverDestroyed<RetainPtr<CGPDFPageRef>> logoPage;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        logoPage.get() = loadPassKitPDFPage(@"PayButtonLogoBlack");
    });
    return logoPage;
}

static RetainPtr<CGPDFPageRef> applePayLogoForStyle(ApplePayLogoStyle style)
{
    switch (style) {
    case ApplePayLogoStyle::White:
        return applePayLogoWhite();

    case ApplePayLogoStyle::Black:
        return applePayLogoBlack();
    }

    ASSERT_NOT_REACHED();
    return nullptr;
}

WTF_MAKE_TZONE_ALLOCATED_IMPL(ApplePayLogoSystemImage);

void ApplePayLogoSystemImage::draw(GraphicsContext& context, const FloatRect& destinationRect) const
{
    auto page = applePayLogoForStyle(m_applePayLogoStyle);
    if (!page)
        return;
    CGContextRef cgContext = context.platformContext();
    CGContextSaveGState(cgContext);
    CGSize pdfSize = CGPDFPageGetBoxRect(page.get(), kCGPDFMediaBox).size;

    auto largestRect = largestRectWithAspectRatioInsideRect(pdfSize.width / pdfSize.height, destinationRect);
    CGContextTranslateCTM(cgContext, largestRect.x(), largestRect.y());
    auto scale = largestRect.width() / pdfSize.width;
    CGContextScaleCTM(cgContext, scale, scale);

    CGContextTranslateCTM(cgContext, 0, pdfSize.height);
    CGContextScaleCTM(cgContext, 1, -1);

    CGContextDrawPDFPage(cgContext, page.get());
    CGContextRestoreGState(cgContext);
}

} // namespace WebCore

#endif // ENABLE(APPLE_PAY)
