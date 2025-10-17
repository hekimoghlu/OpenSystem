/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 4, 2021.
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

#include "GradientAttributes.h"

namespace WebCore {
struct LinearGradientAttributes : GradientAttributes {
    LinearGradientAttributes()
        : m_x1()
        , m_y1()
        , m_x2(SVGLengthMode::Width, "100%"_s)
        , m_y2()
        , m_x1Set(false)
        , m_y1Set(false)
        , m_x2Set(false)
        , m_y2Set(false)
    {
    }

    SVGLengthValue x1() const { return m_x1; }
    SVGLengthValue y1() const { return m_y1; }
    SVGLengthValue x2() const { return m_x2; }
    SVGLengthValue y2() const { return m_y2; }

    void setX1(SVGLengthValue value) { m_x1 = value; m_x1Set = true; }
    void setY1(SVGLengthValue value) { m_y1 = value; m_y1Set = true; }
    void setX2(SVGLengthValue value) { m_x2 = value; m_x2Set = true; }
    void setY2(SVGLengthValue value) { m_y2 = value; m_y2Set = true; }

    bool hasX1() const { return m_x1Set; }
    bool hasY1() const { return m_y1Set; }
    bool hasX2() const { return m_x2Set; }
    bool hasY2() const { return m_y2Set; }

private:
    // Properties
    SVGLengthValue m_x1;
    SVGLengthValue m_y1;
    SVGLengthValue m_x2;
    SVGLengthValue m_y2;

    // Property states
    bool m_x1Set : 1;
    bool m_y1Set : 1;
    bool m_x2Set : 1;
    bool m_y2Set : 1;
};

} // namespace WebCore
