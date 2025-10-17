/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, June 8, 2025.
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

#include "ExceptionOr.h"
#include "QualifiedName.h"
#include "SVGNames.h"
#include "SVGZoomAndPanType.h"

namespace WebCore {

class SVGZoomAndPan {
    WTF_MAKE_NONCOPYABLE(SVGZoomAndPan);
public:
    // Forward declare enumerations in the W3C naming scheme, for IDL generation.
    enum {
        SVG_ZOOMANDPAN_UNKNOWN = SVGZoomAndPanUnknown,
        SVG_ZOOMANDPAN_DISABLE = SVGZoomAndPanDisable,
        SVG_ZOOMANDPAN_MAGNIFY = SVGZoomAndPanMagnify
    };

    SVGZoomAndPanType zoomAndPan() const { return m_zoomAndPan; }
    void setZoomAndPan(SVGZoomAndPanType zoomAndPan) { m_zoomAndPan = zoomAndPan; }
    ExceptionOr<void> setZoomAndPan(unsigned) { return Exception { ExceptionCode::NoModificationAllowedError }; }
    void reset() { m_zoomAndPan = SVGPropertyTraits<SVGZoomAndPanType>::initialValue(); }

    void parseAttribute(const QualifiedName&, const AtomString&);

protected:
    SVGZoomAndPan() = default;

    static std::optional<SVGZoomAndPanType> parseZoomAndPan(StringParsingBuffer<LChar>&);
    static std::optional<SVGZoomAndPanType> parseZoomAndPan(StringParsingBuffer<UChar>&);

private:
    SVGZoomAndPanType m_zoomAndPan { SVGPropertyTraits<SVGZoomAndPanType>::initialValue() };
};

} // namespace WebCore
