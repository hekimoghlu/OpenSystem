/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 31, 2022.
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

#include "RenderTreeBuilder.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class RenderBlock;
class RenderButton;
class RenderMenuList;

class RenderTreeBuilder::FormControls {
    WTF_MAKE_TZONE_ALLOCATED(FormControls);
public:
    FormControls(RenderTreeBuilder&);

    void attach(RenderButton& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild);
    void attach(RenderMenuList& parent, RenderPtr<RenderObject> child, RenderObject* beforeChild);

    RenderPtr<RenderObject> detach(RenderButton& parent, RenderObject& child, RenderTreeBuilder::WillBeDestroyed) WARN_UNUSED_RETURN;
    RenderPtr<RenderObject> detach(RenderMenuList& parent, RenderObject& child, RenderTreeBuilder::WillBeDestroyed) WARN_UNUSED_RETURN;

private:
    RenderBlock& findOrCreateParentForChild(RenderButton&);
    RenderBlock& findOrCreateParentForChild(RenderMenuList&);

    RenderTreeBuilder& m_builder;
};

}

