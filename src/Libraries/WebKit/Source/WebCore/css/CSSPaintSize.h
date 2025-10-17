/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 14, 2022.
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

#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class CSSPaintSize : public RefCounted<CSSPaintSize> {
public:
    static Ref<CSSPaintSize> create(double width, double height)
    {
        return adoptRef(*new CSSPaintSize(width, height));
    }

    double width() const { return m_width; }
    double height() const { return m_height; }

private:
    CSSPaintSize(double width, double height)
        : m_width(width)
        , m_height(height)
    {
    }

    double m_width;
    double m_height;
};

} // namespace WebCore
