/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 11, 2023.
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
#include "config.h"
#include "WKGeometry.h"

#include "APIGeometry.h"
#include "WKAPICast.h"

WKTypeID WKSizeGetTypeID(void)
{
    return WebKit::toAPI(API::Size::APIType);
}

WKTypeID WKPointGetTypeID(void)
{
    return WebKit::toAPI(API::Point::APIType);
}

WKTypeID WKRectGetTypeID(void)
{
    return WebKit::toAPI(API::Rect::APIType);
}

WKPointRef WKPointCreate(WKPoint point)
{
    return WebKit::toAPI(&API::Point::create(point).leakRef());
}

WKSizeRef WKSizeCreate(WKSize size)
{
    return WebKit::toAPI(&API::Size::create(size).leakRef());
}

WKRectRef WKRectCreate(WKRect rect)
{
    return WebKit::toAPI(&API::Rect::create(rect).leakRef());
}

WKSize WKSizeGetValue(WKSizeRef size)
{
    return WebKit::toImpl(size)->size();
}

WKPoint WKPointGetValue(WKPointRef point)
{
    return WebKit::toImpl(point)->point();
}

WKRect WKRectGetValue(WKRectRef rect)
{
    return WebKit::toImpl(rect)->rect();
}

