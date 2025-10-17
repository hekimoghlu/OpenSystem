/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 12, 2024.
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

#include "FormattingConstraints.h"
#include "LayoutElementBox.h"
#include "LayoutUnit.h"
#include "LayoutUnits.h"
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class LayoutSize;
struct Length;

namespace Layout {

class BoxGeometry;
class ElementBox;
struct ConstraintsForInFlowContent;
class FormattingGeometry;
class FormattingQuirks;
struct IntrinsicWidthConstraints;
class LayoutState;

class FormattingContext {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(FormattingContext);
public:
    virtual ~FormattingContext();

    virtual void layoutInFlowContent(const ConstraintsForInFlowContent&) { };
    virtual IntrinsicWidthConstraints computedIntrinsicWidthConstraints() { return { }; }
    virtual LayoutUnit usedContentHeight() const { return { }; }

    const ElementBox& root() const { return m_root; }

    LayoutState& layoutState();
    const LayoutState& layoutState() const { return const_cast<FormattingContext&>(*this).layoutState(); }

    enum class EscapeReason {
        TableQuirkNeedsGeometryFromEstablishedFormattingContext,
        OutOfFlowBoxNeedsInFlowGeometry,
        FloatBoxIsAlwaysRelativeToFloatStateRoot,
        FindFixedHeightAncestorQuirk,
        DocumentBoxStretchesToViewportQuirk,
        BodyStretchesToViewportQuirk,
        TableNeedsAccessToTableWrapper
    };
    const BoxGeometry& geometryForBox(const Box&, std::optional<EscapeReason> = std::nullopt) const;
    BoxGeometry& geometryForBox(const Box&, std::optional<EscapeReason> = std::nullopt);

    bool isBlockFormattingContext() const { return root().establishesBlockFormattingContext(); }
    bool isTableFormattingContext() const { return root().establishesTableFormattingContext(); }
    bool isTableWrapperBlockFormattingContext() const { return isBlockFormattingContext() && root().isTableWrapperBox(); }
    bool isFlexFormattingContext() const { return root().establishesFlexFormattingContext(); }

    static const InitialContainingBlock& initialContainingBlock(const Box&);
    static const ElementBox& containingBlock(const Box&);
#if ASSERT_ENABLED
    static const ElementBox& formattingContextRoot(const Box&);
#endif

protected:
    FormattingContext(const ElementBox& formattingContextRoot, LayoutState&);

#if ASSERT_ENABLED
    virtual void validateGeometryConstraintsAfterLayout() const;
#endif

private:
    CheckedRef<const ElementBox> m_root;
    LayoutState& m_layoutState;
};

}
}

#define SPECIALIZE_TYPE_TRAITS_LAYOUT_FORMATTING_CONTEXT(ToValueTypeName, predicate) \
SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::Layout::ToValueTypeName) \
    static bool isType(const WebCore::Layout::FormattingContext& formattingContext) { return formattingContext.predicate; } \
SPECIALIZE_TYPE_TRAITS_END()

