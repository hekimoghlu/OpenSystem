/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 10, 2025.
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
#if PLATFORM(MAC)

#import "Connection.h"
#import <WebCore/IntRectHash.h>
#import <condition_variable>
#import <wtf/Condition.h>
#import <wtf/HashMap.h>
#import <wtf/Lock.h>
#import <wtf/RetainPtr.h>
#import <wtf/Vector.h>
#import <wtf/WeakObjCPtr.h>

@class WKPrintingViewData;
@class PDFDestination;
@class PDFDocument;

namespace WebCore {
class ShareableBitmap;
}

namespace WebKit {
class WebFrameProxy;
}

@interface WKPrintingView : NSView {
@public
    WeakObjCPtr<NSPrintOperation> _printOperation;
    RetainPtr<NSView> _wkView;

    RefPtr<WebKit::WebFrameProxy> _webFrame;
    Vector<WebCore::IntRect> _printingPageRects;
    double _totalScaleFactorForPrinting;
    HashMap<WebCore::IntRect, RefPtr<WebCore::ShareableBitmap>> _pagePreviews;

    Vector<uint8_t> _printedPagesData;
    RetainPtr<PDFDocument> _printedPagesPDFDocument;
    Vector<Vector<RetainPtr<PDFDestination>>> _linkDestinationsPerPage;

    Markable<IPC::Connection::AsyncReplyID> _expectedComputedPagesCallback;
    HashMap<IPC::Connection::AsyncReplyID, WebCore::IntRect> _expectedPreviewCallbacks;
    Markable<IPC::Connection::AsyncReplyID> _latestExpectedPreviewCallback;
    Markable<IPC::Connection::AsyncReplyID> _expectedPrintCallback;

    BOOL _isPrintingFromSecondaryThread;
    Lock _printingCallbackMutex;
    Condition _printingCallbackCondition;

    NSTimer *_autodisplayResumeTimer;
}

- (id)initWithFrameProxy:(WebKit::WebFrameProxy&)frame view:(NSView *)wkView;

@end

#endif // PLATFORM(MAC)
