/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 12, 2023.
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
#import "_WKWebViewPrintFormatter.h"

#if PLATFORM(IOS_FAMILY)

@interface UIPrintFormatter ()
- (CGRect)_pageContentRect:(BOOL)firstPage;
- (void)_setNeedsRecalc;
@end

@interface _WKWebViewPrintFormatter ()
- (BOOL)_shouldDrawUsingBitmap;
- (void)_setSnapshotPaperRect:(CGRect)paperRect;
- (void)_setPrintedDocument:(CGPDFDocumentRef)printedDocument;
- (void)_setPrintPreviewImage:(CGImageRef)printPreviewImage;
- (void)_invalidatePrintRenderingState;

- (void)_waitForPrintedDocumentOrImage;
@end

@protocol _WKWebViewPrintProvider <NSObject>

@property (nonatomic, readonly) BOOL _wk_printFormatterRequiresMainThread;

- (NSUInteger)_wk_pageCountForPrintFormatter:(_WKWebViewPrintFormatter *)printFormatter;
- (void)_wk_requestDocumentForPrintFormatter:(_WKWebViewPrintFormatter *)printFormatter;

@optional
- (void)_wk_requestImageForPrintFormatter:(_WKWebViewPrintFormatter *)printFormatter;
@end

#endif // PLATFORM(IOS_FAMILY)
