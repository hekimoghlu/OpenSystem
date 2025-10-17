/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, March 1, 2025.
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

#include "RenderBlockFlow.h"

namespace WebCore {

class HTMLInputElement;

// Each RenderFileUploadControl contains a RenderButton (for opening the file chooser), and
// sufficient space to draw a file icon and filename. The RenderButton has a shadow node
// associated with it to receive click/hover events.

class RenderFileUploadControl final : public RenderBlockFlow {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(RenderFileUploadControl);
    WTF_OVERRIDE_DELETE_FOR_CHECKED_PTR(RenderFileUploadControl);
public:
    RenderFileUploadControl(HTMLInputElement&, RenderStyle&&);
    virtual ~RenderFileUploadControl();

    String buttonValue();
    String fileTextValue() const;

    HTMLInputElement& inputElement() const;
    
private:
    void element() const = delete;

    ASCIILiteral renderName() const override { return "RenderFileUploadControl"_s; }

    void updateFromElement() override;
    void computeIntrinsicLogicalWidths(LayoutUnit& minLogicalWidth, LayoutUnit& maxLogicalWidth) const override;
    void computePreferredLogicalWidths() override;
    void paintObject(PaintInfo&, const LayoutPoint&) override;
    void paintControl(PaintInfo&, const LayoutPoint&);

    int maxFilenameLogicalWidth() const;

    VisiblePosition positionForPoint(const LayoutPoint&, HitTestSource, const RenderFragmentContainer*) override;

    HTMLInputElement* uploadButton() const;

    bool m_canReceiveDroppedFiles;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_RENDER_OBJECT(RenderFileUploadControl, isRenderFileUploadControl())
