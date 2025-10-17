/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 25, 2023.
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

#import <PDFKit/PDFKit.h>
#import <pal/spi/cg/CoreGraphicsSPI.h>

#if USE(APPLE_INTERNAL_SDK)

#if HAVE(PDFKIT)

#if PLATFORM(IOS_FAMILY)
#import <PDFKit/PDFHostViewController.h>
#endif // PLATFORM(IOS_FAMILY)

#import <PDFKit/PDFDocumentPriv.h>
#import <PDFKit/PDFPagePriv.h>
#import <PDFKit/PDFSelectionPriv.h>

#if HAVE(PDFKIT_WITH_NEXT_ACTIONS)
#import <PDFKit/PDFActionPriv.h>
#endif

#endif // HAVE(PDFKIT)

#else

#if HAVE(PDFKIT)

#if PLATFORM(IOS_FAMILY)
#import "UIKitSPI.h"

@interface _UIRemoteViewController : UIViewController
@end

@protocol PDFHostViewControllerDelegate<NSObject>
@end

@interface PDFHostViewController : _UIRemoteViewController<UIGestureRecognizerDelegate, UIDocumentPasswordViewDelegate>

@property (nonatomic, class) bool useIOSurfaceForTiles;

+ (void) createHostView:(void(^)(PDFHostViewController* hostViewController)) callback forExtensionIdentifier:(NSString*) extensionIdentifier;
- (void) setDelegate:(id<PDFHostViewControllerDelegate>) delegate;
- (void) setDocumentData:(NSData*) data withScrollView:(UIScrollView*) scrollView;

- (void) findString:(NSString*) string withOptions:(NSStringCompareOptions) options;
- (void) cancelFindString;
- (void) cancelFindStringWithHighlightsCleared:(BOOL)cleared;
- (void) focusOnSearchResultAtIndex:(NSUInteger) searchIndex;

- (NSInteger) currentPageIndex;
- (NSInteger) pageCount;
- (UIView*) pageNumberIndicator;
- (void) goToPageIndex:(NSInteger) pageIndex;
- (void) updatePDFViewLayout;

+ (UIColor *)backgroundColor;

- (void) beginPDFViewRotation;
- (void) endPDFViewRotation;

- (void) snapshotViewRect: (CGRect) rect snapshotWidth: (NSNumber*) width afterScreenUpdates: (BOOL) afterScreenUpdates withResult: (void (^)(UIImage* image)) completion;

@end
#endif // PLATFORM(IOS_FAMILY)

@interface PDFSelection (SPI)
- (void)drawForPage:(PDFPage *)page withBox:(CGPDFBox)box active:(BOOL)active inContext:(CGContextRef)context;
- (PDFPoint)firstCharCenter;
- (/*nullable*/ NSString *)html;
- (BOOL)isEmpty;
#if HAVE(PDFSELECTION_ENUMERATE_RECTS_AND_TRANSFORMS)
- (void)enumerateRectsAndTransformsForPage:(PDFPage *)page usingBlock:(void (^)(CGRect rect, CGAffineTransform transform))block;
#endif
@end

@interface PDFDocument (Annotations)
#if HAVE(PDFDOCUMENT_RESET_FORM_FIELDS)
- (void)resetFormFields:(PDFActionResetForm *)action;
#endif
#if HAVE(PDFDOCUMENT_ANNOTATIONS_FOR_FIELD_NAME)
- (NSArray *)annotationsForFieldName:(NSString *)fieldname;
#endif
@end

@interface PDFAction (PDFActionPriv)
- (NSArray *)nextActions;
@end

#if HAVE(INCREMENTAL_PDF_APIS)
@interface PDFDocument (IncrementalLoading)
-(instancetype)initWithProvider:(CGDataProviderRef)dataProvider;
-(void)preloadDataOfPagesInRange:(NSRange)range onQueue:(dispatch_queue_t)queue completion:(void (^)(NSIndexSet* loadedPageIndexes))completionBlock;
@property (readwrite, nonatomic) BOOL hasHighLatencyDataProvider;
@end
#endif // HAVE(INCREMENTAL_PDF_APIS)

#endif // HAVE(PDFKIT)

#endif // USE(APPLE_INTERNAL_SDK)

#if ENABLE(UNIFIED_PDF)
@interface PDFDocument (IPI)
- (PDFDestination *)namedDestination:(NSString *)name;
@end

#if HAVE(COREGRAPHICS_WITH_PDF_AREA_OF_INTEREST_SUPPORT)
@interface PDFPage (IPI)
- (CGPDFPageLayoutRef) pageLayout;
@end
#endif

#if HAVE(PDFPAGE_AREA_OF_INTEREST_AT_POINT)
#define PDFAreaOfInterest NSInteger

#define kPDFTextArea        (1UL << 1)
#define kPDFAnnotationArea  (1UL << 2)
#define kPDFLinkArea        (1UL << 3)
#define kPDFControlArea     (1UL << 4)
#define kPDFTextFieldArea   (1UL << 5)
#define kPDFIconArea        (1UL << 6)
#define kPDFPopupArea       (1UL << 7)
#define kPDFImageArea       (1UL << 8)

@interface PDFPage (Staging_119217538)
- (PDFAreaOfInterest)areaOfInterestAtPoint:(PDFPoint)point;
@end
#endif

#if ENABLE(UNIFIED_PDF_DATA_DETECTION)

#if HAVE(PDFDOCUMENT_ENABLE_DATA_DETECTORS)
@interface PDFDocument (Staging_123761050)
@property (nonatomic) BOOL enableDataDetectors;
@end
#endif

#if HAVE(PDFPAGE_DATA_DETECTOR_RESULTS)
@interface PDFPage (Staging_123761050)
- (NSArray *)dataDetectorResults;
@end
#endif

#endif

#if HAVE(PDFSELECTION_HTMLDATA_RTFDATA)

@interface PDFSelection (Staging_136075998)
- (/*nullable*/ NSData *)htmlData;
- (/*nullable*/ NSData *)rtfData;
@end

#endif

#endif // ENABLE(UNIFIED_PDF)

// FIXME: Move this declaration inside the !USE(APPLE_INTERNAL_SDK) block once rdar://problem/118903435 is in builds.
@interface PDFDocument (AX)
- (NSArray *)accessibilityChildren:(id)parent;
@end

@interface PDFAnnotation (AccessibilityPrivate)
- (id)accessibilityNode;
@end
