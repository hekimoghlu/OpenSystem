/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 6, 2023.
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
#ifndef WKImage_h
#define WKImage_h

#include <WebKit/WKBase.h>
#include <WebKit/WKGeometry.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    kWKImageOptionsShareable = 1 << 0,
};
typedef uint32_t WKImageOptions;

enum {
    kWKSnapshotOptionsShareable = 1 << 0,
    kWKSnapshotOptionsExcludeSelectionHighlighting = 1 << 1,
    kWKSnapshotOptionsInViewCoordinates = 1 << 2,
    kWKSnapshotOptionsPaintSelectionRectangle = 1 << 3,
    kWKSnapshotOptionsForceBlackText = 1 << 4,
    kWKSnapshotOptionsForceWhiteText = 1 << 5,
    kWKSnapshotOptionsPrinting = 1 << 6,
    kWKSnapshotOptionsExcludeOverflow = 1 << 7,
    kWKSnapshotOptionsExtendedColor = 1 << 8,
};
typedef uint32_t WKSnapshotOptions;

WK_EXPORT WKTypeID WKImageGetTypeID(void);

WK_EXPORT WKImageRef WKImageCreate(WKSize size, WKImageOptions options);

WK_EXPORT WKSize WKImageGetSize(WKImageRef image);

#ifdef __cplusplus
}
#endif

#endif /* WKImage_h */
