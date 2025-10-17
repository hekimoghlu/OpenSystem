/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 17, 2023.
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
#import <WebKit/WKFoundation.h>

typedef NS_OPTIONS(NSUInteger, _WKRenderingProgressEvents) {
    _WKRenderingProgressEventFirstLayout = 1 << 0,
    _WKRenderingProgressEventFirstVisuallyNonEmptyLayout WK_API_AVAILABLE(macos(10.11), ios(9.0)) = 1 << 1,
    _WKRenderingProgressEventFirstPaintWithSignificantArea = 1 << 2,
    _WKRenderingProgressEventReachedSessionRestorationRenderTreeSizeThreshold WK_API_AVAILABLE(macos(10.11), ios(9.0)) = 1 << 3,
    _WKRenderingProgressEventFirstLayoutAfterSuppressedIncrementalRendering WK_API_AVAILABLE(macos(10.11), ios(9.0)) = 1 << 4,
    _WKRenderingProgressEventFirstPaintAfterSuppressedIncrementalRendering WK_API_AVAILABLE(macos(10.11), ios(9.0)) = 1 << 5,
    _WKRenderingProgressEventFirstPaint WK_API_AVAILABLE(macos(10.11), ios(9.0)) = 1 << 6,
    _WKRenderingProgressEventDidRenderSignificantAmountOfText WK_API_AVAILABLE(macos(10.14), ios(12.0)) = 1 << 7,
    _WKRenderingProgressEventFirstMeaningfulPaint WK_API_AVAILABLE(macos(10.14.4), ios(12.2)) = 1 << 8,
} WK_API_AVAILABLE(macos(10.10), ios(8.0));
