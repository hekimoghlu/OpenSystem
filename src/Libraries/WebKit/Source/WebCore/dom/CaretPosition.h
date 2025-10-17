/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 30, 2023.
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

#include "ScriptWrappable.h"

namespace WebCore {

class DOMRect;
class Node;

class CaretPosition : public ScriptWrappable, public RefCounted<CaretPosition> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED_EXPORT(CaretPosition, WEBCORE_EXPORT);
public:
    static Ref<CaretPosition> create(RefPtr<Node>&& offsetNode, unsigned offset) { return adoptRef(*new CaretPosition(WTFMove(offsetNode), offset)); }

    RefPtr<Node> offsetNode() const { return m_offsetNode; }
    unsigned offset() const { return m_offset; }

    RefPtr<DOMRect> getClientRect();

private:
    CaretPosition(RefPtr<Node>&& offsetNode, unsigned offset);

    RefPtr<Node> m_offsetNode;
    unsigned m_offset;
};

} // namespace WebCore
