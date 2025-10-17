/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 15, 2025.
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

#include "LayoutElementBox.h"
#include "SecurityOrigin.h"
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
namespace Layout {
class LayoutState;
}
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::Layout::LayoutState> : std::true_type { };
}

namespace WebCore {

class Document;

namespace LayoutIntegration {
enum class LogicalWidthType : uint8_t;
enum class LogicalHeightType : uint8_t;
}

namespace Layout {

class BlockFormattingState;
class BoxGeometry;
class FormattingContext;
class FormattingState;
class InlineContentCache;
class TableFormattingState;

class LayoutState : public CanMakeWeakPtr<LayoutState> {
    WTF_MAKE_NONCOPYABLE(LayoutState);
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(LayoutState);
public:
    // Primary layout state has a direct geometry cache in layout boxes.
    enum class Type { Primary, Secondary };

    using FormattingContextLayoutFunction = Function<void(const ElementBox&, std::optional<LayoutUnit>, LayoutState&)>;
    using FormattingContextLogicalWidthFunction = Function<LayoutUnit(const ElementBox&, LayoutIntegration::LogicalWidthType)>;
    using FormattingContextLogicalHeightFunction = Function<LayoutUnit(const ElementBox&, LayoutIntegration::LogicalHeightType)>;

    LayoutState(const Document&, const ElementBox& rootContainer, Type, FormattingContextLayoutFunction&&, FormattingContextLogicalWidthFunction&&, FormattingContextLogicalHeightFunction&&);
    ~LayoutState();

    Type type() const { return m_type; }

    void updateQuirksMode(const Document&);

    InlineContentCache& inlineContentCache(const ElementBox& formattingContextRoot);

    BlockFormattingState& ensureBlockFormattingState(const ElementBox& formattingContextRoot);
    TableFormattingState& ensureTableFormattingState(const ElementBox& formattingContextRoot);

    BlockFormattingState& formattingStateForBlockFormattingContext(const ElementBox& blockFormattingContextRoot) const;
    TableFormattingState& formattingStateForTableFormattingContext(const ElementBox& tableFormattingContextRoot) const;

    FormattingState& formattingStateForFormattingContext(const ElementBox& formattingRoot) const;

    void destroyBlockFormattingState(const ElementBox& formattingContextRoot);
    void destroyInlineContentCache(const ElementBox& formattingContextRoot);

    bool hasFormattingState(const ElementBox& formattingRoot) const;

#ifndef NDEBUG
    void registerFormattingContext(const FormattingContext&);
    void deregisterFormattingContext(const FormattingContext& formattingContext) { m_formattingContextList.remove(&formattingContext); }
#endif

    BoxGeometry& geometryForRootBox();
    BoxGeometry& ensureGeometryForBox(const Box&);
    const BoxGeometry& geometryForBox(const Box&) const;

    bool hasBoxGeometry(const Box&) const;

    enum class QuirksMode { No, Limited, Yes };
    bool inQuirksMode() const { return m_quirksMode == QuirksMode::Yes; }
    bool inLimitedQuirksMode() const { return m_quirksMode == QuirksMode::Limited; }
    bool inStandardsMode() const { return m_quirksMode == QuirksMode::No; }
    const SecurityOrigin& securityOrigin() const { return m_securityOrigin.get(); }

    const ElementBox& root() const { return m_rootContainer; }

    void layoutWithFormattingContextForBox(const ElementBox&, std::optional<LayoutUnit> widthConstraint) const;
    LayoutUnit logicalWidthWithFormattingContextForBox(const ElementBox&, LayoutIntegration::LogicalWidthType) const;
    LayoutUnit logicalHeightWithFormattingContextForBox(const ElementBox&, LayoutIntegration::LogicalHeightType) const;

private:
    void setQuirksMode(QuirksMode quirksMode) { m_quirksMode = quirksMode; }
    BoxGeometry& ensureGeometryForBoxSlow(const Box&);

    const Type m_type;

    UncheckedKeyHashMap<const ElementBox*, std::unique_ptr<InlineContentCache>> m_inlineContentCaches;

    UncheckedKeyHashMap<const ElementBox*, std::unique_ptr<BlockFormattingState>> m_blockFormattingStates;
    UncheckedKeyHashMap<const ElementBox*, std::unique_ptr<TableFormattingState>> m_tableFormattingStates;

#ifndef NDEBUG
    UncheckedKeyHashSet<const FormattingContext*> m_formattingContextList;
#endif
    UncheckedKeyHashMap<const Box*, std::unique_ptr<BoxGeometry>> m_layoutBoxToBoxGeometry;
    QuirksMode m_quirksMode { QuirksMode::No };

    CheckedRef<const ElementBox> m_rootContainer;
    Ref<SecurityOrigin> m_securityOrigin;

    FormattingContextLayoutFunction m_formattingContextLayoutFunction;
    FormattingContextLogicalWidthFunction m_formattingContextLogicalWidthFunction;
    FormattingContextLogicalHeightFunction m_formattingContextLogicalHeightFunction;
};

inline bool LayoutState::hasBoxGeometry(const Box& layoutBox) const
{
    if (LIKELY(m_type == Type::Primary))
        return !!layoutBox.m_cachedGeometryForPrimaryLayoutState;

    return m_layoutBoxToBoxGeometry.contains(&layoutBox);
}

inline BoxGeometry& LayoutState::ensureGeometryForBox(const Box& layoutBox)
{
    if (LIKELY(m_type == Type::Primary)) {
        if (auto* boxGeometry = layoutBox.m_cachedGeometryForPrimaryLayoutState.get()) {
            ASSERT(layoutBox.m_primaryLayoutState == this);
            return *boxGeometry;
        }
    }
    return ensureGeometryForBoxSlow(layoutBox);
}

inline const BoxGeometry& LayoutState::geometryForBox(const Box& layoutBox) const
{
    if (LIKELY(m_type == Type::Primary)) {
        ASSERT(layoutBox.m_primaryLayoutState == this);
        return *layoutBox.m_cachedGeometryForPrimaryLayoutState;
    }

    ASSERT(layoutBox.m_primaryLayoutState != this);
    ASSERT(m_layoutBoxToBoxGeometry.contains(&layoutBox));
    return *m_layoutBoxToBoxGeometry.get(&layoutBox);
}

#ifndef NDEBUG
inline void LayoutState::registerFormattingContext(const FormattingContext& formattingContext)
{
    // Multiple formatting contexts of the same root within a layout frame indicates defective layout logic.
    ASSERT(!m_formattingContextList.contains(&formattingContext));
    m_formattingContextList.add(&formattingContext);
}
#endif

}
}
