/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 30, 2022.
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

#include "CSSPathFunction.h"
#include "CSSValue.h"

namespace WebCore {

// `CSSPathValue` is used to represent a path for the `d` property from SVG.
// https://svgwg.org/svg2-draft/paths.html#DProperty
class CSSPathValue final : public CSSValue {
public:
    static Ref<CSSPathValue> create(CSS::PathFunction path)
    {
        return adoptRef(*new CSSPathValue(WTFMove(path)));
    }

    const CSS::PathFunction& path() const { return m_path; }

    String customCSSText() const;
    bool equals(const CSSPathValue&) const;

    IterationStatus customVisitChildren(const Function<IterationStatus(CSSValue&)>&) const;

private:
    CSSPathValue(CSS::PathFunction&& path)
        : CSSValue(ClassType::Path)
        , m_path { WTFMove(path) }
    {
    }

    CSS::PathFunction m_path;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_CSS_VALUE(CSSPathValue, isPath())
