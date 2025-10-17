/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 26, 2023.
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
#ifndef WKGeometry_h
#define WKGeometry_h

#include <WebKit/WKBase.h>

#ifdef __cplusplus
extern "C" {
#endif

struct WKPoint {
    double x;
    double y;
};
typedef struct WKPoint WKPoint;

WK_INLINE WKPoint WKPointMake(double x, double y)
{
    WKPoint point;
    point.x = x;
    point.y = y;
    return point;
}

struct WKSize {
    double width;
    double height;
};
typedef struct WKSize WKSize;

WK_INLINE WKSize WKSizeMake(double width, double height)
{
    WKSize size;
    size.width = width;
    size.height = height;
    return size;
}

struct WKRect {
    WKPoint origin;
    WKSize size;
};
typedef struct WKRect WKRect;

WK_INLINE WKRect WKRectMake(double x, double y, double width, double height)
{
    WKRect rect;
    rect.origin.x = x;
    rect.origin.y = y;
    rect.size.width = width;
    rect.size.height = height;
    return rect;
}

WK_EXPORT WKTypeID WKSizeGetTypeID(void);
WK_EXPORT WKTypeID WKPointGetTypeID(void);
WK_EXPORT WKTypeID WKRectGetTypeID(void);

WK_EXPORT WKPointRef WKPointCreate(WKPoint point);
WK_EXPORT WKSizeRef WKSizeCreate(WKSize size);
WK_EXPORT WKRectRef WKRectCreate(WKRect rect);

WK_EXPORT WKSize WKSizeGetValue(WKSizeRef size);
WK_EXPORT WKPoint WKPointGetValue(WKPointRef point);
WK_EXPORT WKRect WKRectGetValue(WKRectRef rect);


#ifdef __cplusplus
}

inline bool operator==(const WKPoint& a, const WKPoint& b)
{
    return a.x == b.x && a.y == b.y;
}
#endif

#endif /* WKGeometry_h */
