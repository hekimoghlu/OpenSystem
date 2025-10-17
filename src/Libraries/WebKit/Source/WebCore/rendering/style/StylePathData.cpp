/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 10, 2023.
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
#include "StylePathData.h"

#include "StyleBuilderState.h"
#include "StylePrimitiveNumericTypes+Blending.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/TextStream.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(StylePathData);

Ref<StylePathData> StylePathData::clone() const
{
    return StylePathData::create(m_path);
}

Path StylePathData::path(const FloatRect& rect) const
{
    return Style::path(m_path, rect);
}

WindRule StylePathData::windRule() const
{
    return Style::windRule(m_path);
}

bool StylePathData::canBlend(const StylePathData& other) const
{
    return Style::canBlend(m_path, other.m_path);
}

Ref<StylePathData> StylePathData::blend(const StylePathData& other, const BlendingContext& context) const
{
    return StylePathData::create(Style::blend(m_path, other.m_path, context));
}

bool StylePathData::operator==(const StylePathData& other) const
{
    return m_path == other.m_path;
}

} // namespace WebCore
