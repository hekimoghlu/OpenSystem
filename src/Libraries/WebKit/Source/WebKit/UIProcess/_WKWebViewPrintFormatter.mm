/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 18, 2022.
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
#import "_WKWebViewPrintFormatterInternal.h"

#if PLATFORM(IOS_FAMILY)

#import "UIKitSPI.h"
#import "WKWebViewInternal.h"
#import "_WKFrameHandle.h"
#import <wtf/Condition.h>
#import <wtf/Locker.h>
#import <wtf/RetainPtr.h>
#import <wtf/SetForScope.h>

@interface UIPrintPageRenderer ()
@property (nonatomic) CGRect paperRect;
@property (nonatomic) CGRect printableRect;
@end

@implementation _WKWebViewPrintFormatter {
    RetainPtr<_WKFrameHandle> _frameToPrint;
    BOOL _suppressPageCountRecalc;

    Lock _printLock;
    Condition _printCompletionCondition;
    RetainPtr<CGPDFDocumentRef> _printedDocument;
    RetainPtr<CGImageRef> _printPreviewImage;
}

- (BOOL)requiresMainThread
{
    return self._webView._printProvider._wk_printFormatterRequiresMainThread;
}

- (_WKFrameHandle *)frameToPrint
{
    return _frameToPrint.get();
}

- (void)setFrameToPrint:(_WKFrameHandle *)frameToPrint
{
    _frameToPrint = frameToPrint;
}

- (WKWebView *)_webView
{
    UIView *view = self.view;
    ASSERT([view isKindOfClass:[WKWebView class]]);
    return static_cast<WKWebView *>(view);
}

- (BOOL)_shouldDrawUsingBitmap
{
    if (self.snapshotFirstPage)
        return NO;

    if (![self._webView._printProvider respondsToSelector:@selector(_wk_requestImageForPrintFormatter:)])
        return NO;

    if (self.printPageRenderer.requestedRenderingQuality == UIPrintRenderingQualityBest)
        return NO;

    return YES;
}

- (CGPDFDocumentRef)_printedDocument
{
    if (self.requiresMainThread)
        return _printedDocument.get();

    Locker locker { _printLock };
    return _printedDocument.get();
}

- (void)_setPrintedDocument:(CGPDFDocumentRef)printedDocument
{
    if (self.requiresMainThread) {
        _printedDocument = printedDocument;
        return;
    }

    Locker locker { _printLock };
    _printedDocument = printedDocument;
    _printCompletionCondition.notifyOne();
}

- (CGImageRef)_printPreviewImage
{
    if (self.requiresMainThread)
        return _printPreviewImage.get();

    Locker locker { _printLock };
    return _printPreviewImage.get();
}

- (void)_setPrintPreviewImage:(CGImageRef)printPreviewImage
{
    if (self.requiresMainThread) {
        _printPreviewImage = printPreviewImage;
        return;
    }

    Locker locker { _printLock };
    _printPreviewImage = printPreviewImage;
    _printCompletionCondition.notifyOne();
}

- (void)_waitForPrintedDocumentOrImage
{
    Locker locker { _printLock };
    _printCompletionCondition.wait(_printLock);
}

- (void)_setSnapshotPaperRect:(CGRect)paperRect
{
    SetForScope suppressPageCountRecalc(_suppressPageCountRecalc, YES);
    UIPrintPageRenderer *printPageRenderer = self.printPageRenderer;
    printPageRenderer.paperRect = paperRect;
    printPageRenderer.printableRect = paperRect;
}

- (void)_invalidatePrintRenderingState
{
    [self _setPrintPreviewImage:nullptr];
    [self _setPrintedDocument:nullptr];
}

- (NSInteger)_recalcPageCount
{
    [self _invalidatePrintRenderingState];
    NSUInteger pageCount = [self._webView._printProvider _wk_pageCountForPrintFormatter:self];
    RELEASE_LOG(Printing, "Recalculated page count. Page count = %zu", pageCount);
    return std::min<NSUInteger>(pageCount, NSIntegerMax);
}

- (void)_setNeedsRecalc
{
    if (!_suppressPageCountRecalc)
        [super _setNeedsRecalc];
}

- (CGRect)rectForPageAtIndex:(NSInteger)pageIndex
{
    if (self.snapshotFirstPage)
        return self.printPageRenderer.paperRect;
    return [self _pageContentRect:pageIndex == self.startPage];
}

- (void)drawInRect:(CGRect)rect forPageAtIndex:(NSInteger)pageIndex
{
    if ([self _shouldDrawUsingBitmap])
        [self _drawInRectUsingBitmap:rect forPageAtIndex:pageIndex];
    else
        [self _drawInRectUsingPDF:rect forPageAtIndex:pageIndex];
}

- (void)_drawInRectUsingBitmap:(CGRect)rect forPageAtIndex:(NSInteger)pageIndex
{
    RetainPtr printPreviewImage = [self _printPreviewImage];
    if (!printPreviewImage) {
        [self._webView._printProvider _wk_requestImageForPrintFormatter:self];
        printPreviewImage = [self _printPreviewImage];
        if (!printPreviewImage)
            return;
    }

    if (!self.pageCount)
        return;

    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextSaveGState(context);

    CGImageRef documentImage = _printPreviewImage.get();

    CGFloat pageImageWidth = CGImageGetWidth(documentImage);
    CGFloat pageImageHeight = CGImageGetHeight(documentImage) / self.pageCount;

    if (!pageImageWidth || !pageImageHeight) {
        CGContextRestoreGState(context);
        return;
    }

    RetainPtr pageImage = adoptCF(CGImageCreateWithImageInRect(documentImage, CGRectMake(0, pageIndex * pageImageHeight, pageImageWidth, pageImageHeight)));

    CGContextTranslateCTM(context, CGRectGetMinX(rect), CGRectGetMaxY(rect));
    CGContextScaleCTM(context, 1, -1);
    CGContextScaleCTM(context, CGRectGetWidth(rect) / pageImageWidth, CGRectGetHeight(rect) / pageImageHeight);
    CGContextDrawImage(context, CGRectMake(0, 0, pageImageWidth, pageImageHeight), pageImage.get());

    CGContextRestoreGState(context);
}

- (void)_drawInRectUsingPDF:(CGRect)rect forPageAtIndex:(NSInteger)pageIndex
{
    RetainPtr<CGPDFDocumentRef> printedDocument = [self _printedDocument];
    if (!printedDocument) {
        [self._webView._printProvider _wk_requestDocumentForPrintFormatter:self];
        printedDocument = [self _printedDocument];
        if (!printedDocument)
            return;
    }

    NSInteger offsetFromStartPage = pageIndex - self.startPage;
    if (offsetFromStartPage < 0)
        return;

    CGPDFPageRef page = CGPDFDocumentGetPage(printedDocument.get(), offsetFromStartPage + 1);
    if (!page)
        return;

    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextSaveGState(context);

    CGContextTranslateCTM(context, CGRectGetMinX(rect), CGRectGetMaxY(rect));
    CGContextScaleCTM(context, 1, -1);
    CGContextConcatCTM(context, CGPDFPageGetDrawingTransform(page, kCGPDFCropBox, CGRectMake(0, 0, CGRectGetWidth(rect), CGRectGetHeight(rect)), 0, true));
    CGContextClipToRect(context, CGPDFPageGetBoxRect(page, kCGPDFCropBox));
    CGContextDrawPDFPage(context, page);

    CGContextRestoreGState(context);
}

@end

#endif // PLATFORM(IOS_FAMILY)
