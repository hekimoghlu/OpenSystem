/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 3, 2022.
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

#include "RenderTreeUpdater.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class RenderElement;
class RenderObject;
class RenderTable;
class RenderTableCell;
class RenderTableSection;
class RenderTableRow;
class RenderTreeBuilder;

class RenderTreeBuilder::Table {
    WTF_MAKE_TZONE_ALLOCATED(Table);
public:
    Table(RenderTreeBuilder&);

    RenderElement& findOrCreateParentForChild(RenderTableRow& parent, const RenderObject& child, RenderObject*& beforeChild);
    RenderElement& findOrCreateParentForChild(RenderTableSection& parent, const RenderObject& child, RenderObject*& beforeChild);
    RenderElement& findOrCreateParentForChild(RenderTable& parent, const RenderObject& child, RenderObject*& beforeChild);

    void attach(RenderTable& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild);
    void attach(RenderTableSection& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild);
    void attach(RenderTableRow& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild);

    bool childRequiresTable(const RenderElement& parent, const RenderObject& child);

    void collapseAndDestroyAnonymousSiblingCells(const RenderTableCell& willBeDestroyed);
    void collapseAndDestroyAnonymousSiblingRows(const RenderTableRow& willBeDestroyed);

private:
    template <typename Parent, typename Child>
    RenderPtr<RenderObject> collapseAndDetachAnonymousNextSibling(Parent*, Child* previousChild, Child* nextChild);

    RenderTreeBuilder& m_builder;
};

}
