/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, May 16, 2024.
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

#include "SVGElement.h"
#include "SVGNames.h"
#include "SVGURIReference.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {
    
class SVGPathElement;

class SVGMPathElement final : public SVGElement, public SVGURIReference {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SVGMPathElement);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(SVGMPathElement);
public:
    static Ref<SVGMPathElement> create(const QualifiedName&, Document&);

    virtual ~SVGMPathElement();

    RefPtr<SVGPathElement> pathElement();

    void targetPathChanged();

    using PropertyRegistry = SVGPropertyOwnerRegistry<SVGMPathElement, SVGElement, SVGURIReference>;

private:
    SVGMPathElement(const QualifiedName&, Document&);

    void attributeChanged(const QualifiedName&, const AtomString& oldValue, const AtomString& newValue, AttributeModificationReason) final;
    void svgAttributeChanged(const QualifiedName&) final;

    void buildPendingResource() final;
    void clearResourceReferences();
    InsertedIntoAncestorResult insertedIntoAncestor(InsertionType, ContainerNode&) final;
    void removedFromAncestor(RemovalType, ContainerNode&) final;

    bool rendererIsNeeded(const RenderStyle&) final { return false; }
    void didFinishInsertingNode() final;
};

} // namespace WebCore
