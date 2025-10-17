/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 6, 2023.
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

struct RadialGradientAttributes : GradientAttributes {
    RadialGradientAttributes()
        : m_cx(SVGLengthMode::Width, "50%"_s)
        , m_cy(SVGLengthMode::Width, "50%"_s)
        , m_r(SVGLengthMode::Width, "50%"_s)
        , m_cxSet(false)
        , m_cySet(false)
        , m_rSet(false)
        , m_fxSet(false)
        , m_fySet(false) 
        , m_frSet(false)
    {
    }

    SVGLengthValue cx() const { return m_cx; }
    SVGLengthValue cy() const { return m_cy; }
    SVGLengthValue r() const { return m_r; }
    SVGLengthValue fx() const { return m_fx; }
    SVGLengthValue fy() const { return m_fy; }
    SVGLengthValue fr() const { return m_fr; }

    void setCx(SVGLengthValue value) { m_cx = value; m_cxSet = true; }
    void setCy(SVGLengthValue value) { m_cy = value; m_cySet = true; }
    void setR(SVGLengthValue value) { m_r = value; m_rSet = true; }
    void setFx(SVGLengthValue value) { m_fx = value; m_fxSet = true; }
    void setFy(SVGLengthValue value) { m_fy = value; m_fySet = true; }
    void setFr(SVGLengthValue value) { m_fr = value; m_frSet = true; }

    bool hasCx() const { return m_cxSet; }
    bool hasCy() const { return m_cySet; }
    bool hasR() const { return m_rSet; }
    bool hasFx() const { return m_fxSet; }
    bool hasFy() const { return m_fySet; }
    bool hasFr() const { return m_frSet; }

private:
    // Properties
    SVGLengthValue m_cx;
    SVGLengthValue m_cy;
    SVGLengthValue m_r;
    SVGLengthValue m_fx;
    SVGLengthValue m_fy;
    SVGLengthValue m_fr;

    // Property states
    bool m_cxSet : 1;
    bool m_cySet : 1;
    bool m_rSet : 1;
    bool m_fxSet : 1;
    bool m_fySet : 1;
    bool m_frSet : 1;
};

} // namespace WebCore
