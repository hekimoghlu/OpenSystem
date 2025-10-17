/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 20, 2023.
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

#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class TextMetrics : public RefCounted<TextMetrics> {
public:
    static Ref<TextMetrics> create() { return adoptRef(*new TextMetrics); }

    double width() const { return m_width; }
    void setWidth(double w) { m_width = w; }

    double actualBoundingBoxLeft() const { return m_actualBoundingBoxLeft; }
    void setActualBoundingBoxLeft(double value) { m_actualBoundingBoxLeft = value; }

    double actualBoundingBoxRight() const { return m_actualBoundingBoxRight; }
    void setActualBoundingBoxRight(double value) { m_actualBoundingBoxRight = value; }

    double fontBoundingBoxAscent() const { return m_fontBoundingBoxAscent; }
    void setFontBoundingBoxAscent(double value) { m_fontBoundingBoxAscent = value; }

    double fontBoundingBoxDescent() const { return m_fontBoundingBoxDescent; }
    void setFontBoundingBoxDescent(double value) { m_fontBoundingBoxDescent = value; }

    double actualBoundingBoxAscent() const { return m_actualBoundingBoxAscent; }
    void setActualBoundingBoxAscent(double value) { m_actualBoundingBoxAscent = value; }

    double actualBoundingBoxDescent() const { return m_actualBoundingBoxDescent; }
    void setActualBoundingBoxDescent(double value) { m_actualBoundingBoxDescent = value; }

    double emHeightAscent() const { return m_emHeightAscent; }
    void setEmHeightAscent(double value) { m_emHeightAscent = value; }

    double emHeightDescent() const { return m_emHeightDescent; }
    void setEmHeightDescent(double value) { m_emHeightDescent = value; }

    double hangingBaseline() const { return m_hangingBaseline; }
    void setHangingBaseline(double value) { m_hangingBaseline = value; }

    double alphabeticBaseline() const { return m_alphabeticBaseline; }
    void setAlphabeticBaseline(double value) { m_alphabeticBaseline = value; }

    double ideographicBaseline() const { return m_ideographicBaseline; }
    void setIdeographicBaseline(double value) { m_ideographicBaseline = value; }

private:
    double m_width { 0 };
    double m_actualBoundingBoxLeft { 0 };
    double m_actualBoundingBoxRight { 0 };
    double m_fontBoundingBoxAscent { 0 };
    double m_fontBoundingBoxDescent { 0 };
    double m_actualBoundingBoxAscent { 0 };
    double m_actualBoundingBoxDescent { 0 };
    double m_emHeightAscent { 0 };
    double m_emHeightDescent { 0 };
    double m_hangingBaseline { 0 };
    double m_alphabeticBaseline { 0 };
    double m_ideographicBaseline { 0 };
};

} // namespace WebCore
