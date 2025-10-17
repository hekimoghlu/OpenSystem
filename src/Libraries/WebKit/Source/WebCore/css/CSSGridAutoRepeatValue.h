/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 11, 2025.
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

#include "CSSValueList.h"

namespace WebCore {

// CSSGridAutoRepeatValue stores the track sizes and line numbers when the auto-repeat
// syntax is used
//
// Right now the auto-repeat syntax is as follows:
// <auto-repeat>  = repeat( [ auto-fill | auto-fit ], <line-names>? <fixed-size> <line-names>? )
//
// meaning that only one fixed size track is allowed. It could be argued that a different
// class storing two CSSGridLineNamesValue and one CSSValue (for the track size) fits
// better but the CSSWG has left the door open to allow more than one track in the
// future. That's why we're using a list, it's prepared for future changes and it also
// allows us to keep the parsing algorithm almost intact.
class CSSGridAutoRepeatValue final : public CSSValueContainingVector {
public:
    static Ref<CSSGridAutoRepeatValue> create(CSSValueID, CSSValueListBuilder);

    CSSValueID autoRepeatID() const;

    String customCSSText() const;
    bool equals(const CSSGridAutoRepeatValue&) const;

private:
    explicit CSSGridAutoRepeatValue(bool isAutoFit, CSSValueListBuilder);

    bool m_isAutoFit { false };
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSGridAutoRepeatValue, isGridAutoRepeatValue());
