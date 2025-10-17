/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
#include "LayoutSize.h"

#include <wtf/text/TextStream.h>

namespace WebCore {

LayoutSize LayoutSize::constrainedBetween(const LayoutSize& min, const LayoutSize& max) const
{
    return {
        std::max(min.width(), std::min(max.width(), m_width)),
        std::max(min.height(), std::min(max.height(), m_height))
    };
}

TextStream& operator<<(TextStream& ts, const LayoutSize& size)
{
    return ts << "width=" << size.width().toFloat() << " height=" << size.height().toFloat();
}

} // namespace WebCore
