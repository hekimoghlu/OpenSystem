/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 3, 2025.
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
#include "SVGTextLayoutAttributes.h"

#include "RenderSVGInlineText.h"
#include <stdio.h>
#include <wtf/text/CString.h>

namespace WebCore {

SVGTextLayoutAttributes::SVGTextLayoutAttributes(RenderSVGInlineText& context)
    : m_context(context)
{
}

void SVGTextLayoutAttributes::clear()
{
    m_characterDataMap.clear();
    m_textMetricsValues.resize(0);
}

RenderSVGInlineText& SVGTextLayoutAttributes::context() { return m_context.get(); }
const RenderSVGInlineText& SVGTextLayoutAttributes::context() const { return m_context.get(); }
}
