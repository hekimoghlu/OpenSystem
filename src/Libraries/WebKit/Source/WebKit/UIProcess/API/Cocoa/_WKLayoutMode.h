/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 25, 2024.
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

typedef NS_ENUM(NSUInteger, _WKLayoutMode) {
    _WKLayoutModeViewSize = 0,
    _WKLayoutModeFixedSize = 1,

    // Lay out the view with its frame scaled by the inverse viewScale.
    _WKLayoutModeDynamicSizeComputedFromViewScale = 2,

    // Lay out the view at a heuristically-determined size based on the minimum size of the document.
    _WKLayoutModeDynamicSizeComputedFromMinimumDocumentSize = 4,

} WK_API_AVAILABLE(macos(10.11), ios(9.0));
