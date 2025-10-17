/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 14, 2022.
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
#include "CaretPosition.h"

#include "DOMRect.h"
#include "Range.h"
#include "SimpleRange.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(CaretPosition);

CaretPosition::CaretPosition(RefPtr<Node>&& offsetNode, unsigned offset)
    : m_offsetNode(WTFMove(offsetNode))
    , m_offset(offset)
{
}

RefPtr<DOMRect> CaretPosition::getClientRect()
{
    if (!m_offsetNode)
        return nullptr;

    return Range::boundingClientRect(SimpleRange { BoundaryPoint { *m_offsetNode, m_offset }, BoundaryPoint { *m_offsetNode, m_offset } });
}

} // namespace WebCore
