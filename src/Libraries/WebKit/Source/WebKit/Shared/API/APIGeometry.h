/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 30, 2021.
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

#include "APIObject.h"
#include "WKGeometry.h"

namespace API {

class Size : public API::ObjectImpl<API::Object::Type::Size> {
public:
    static Ref<Size> create(const WKSize& size)
    {
        return adoptRef(*new Size(size));
    }
    static Ref<Size> create(double width, double height)
    {
        return create(WKSizeMake(width, height));
    }

    const WKSize& size() const { return m_size; }

private:
    explicit Size(const WKSize& size)
        : m_size(size)
    {
    }

    WKSize m_size;
};

class Point : public API::ObjectImpl<API::Object::Type::Point> {
public:
    static Ref<Point> create(const WKPoint& point)
    {
        return adoptRef(*new Point(point));
    }
    static Ref<Point> create(double x, double y)
    {
        return adoptRef(*new Point(WKPointMake(x, y)));
    }

    const WKPoint& point() const { return m_point; }

private:
    explicit Point(const WKPoint& point)
        : m_point(point)
    { }

    WKPoint m_point;
};

class Rect : public API::ObjectImpl<API::Object::Type::Rect> {
public:
    static Ref<Rect> create(const WKRect& rect)
    {
        return adoptRef(*new Rect(rect));
    }
    static Ref<Rect> create(double x, double y, double width, double height)
    {
        return create(WKRectMake(x, y, width, height));
    }

    const WKRect& rect() const { return m_rect; }

private:
    explicit Rect(const WKRect& rect)
        : m_rect(rect)
    {
    }

    WKRect m_rect;
};

} // namespace API

SPECIALIZE_TYPE_TRAITS_API_OBJECT(Size);
SPECIALIZE_TYPE_TRAITS_API_OBJECT(Point);
SPECIALIZE_TYPE_TRAITS_API_OBJECT(Rect);
