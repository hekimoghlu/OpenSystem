/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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

#include "RenderFragmentContainer.h"

namespace WebCore {

class RenderFragmentedFlow;

// RenderFragmentContainerSet represents a set of regions that all have the same width and height. It is a "composite region box" that
// can be used to represent a single run of contiguous regions.
//
// By combining runs of same-size columns or pages into a single object, we significantly reduce the number of unique RenderObjects
// required to represent those objects.
//
// This class is abstract and is only intended for use by renderers that generate anonymous runs of identical regions, i.e.,
// columns and printing. RenderMultiColumnSet and RenderPageSet represent runs of columns and pages respectively.
//
// FIXME: For now we derive from RenderFragmentContainer, but this may change at some point.

class RenderFragmentContainerSet : public RenderFragmentContainer {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderFragmentContainerSet);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderFragmentContainerSet);
public:
    void expandToEncompassFragmentedFlowContentsIfNeeded();

protected:
    RenderFragmentContainerSet(Type, Document&, RenderStyle&&, RenderFragmentedFlow&);
    virtual ~RenderFragmentContainerSet();

private:
    void installFragmentedFlow() final;

    ASCIILiteral renderName() const override = 0;
    
    bool isRenderFragmentContainerSet() const final { return true; }
};

} // namespace WebCore
