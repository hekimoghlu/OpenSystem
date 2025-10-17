/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 10, 2024.
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

#include "CSSSelectorList.h"
#include "CSSSelectorParser.h"
#include "ExceptionOr.h"
#include "NodeList.h"
#include "SecurityOriginData.h"
#include "SelectorCompiler.h"
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/Vector.h>
#include <wtf/text/AtomStringHash.h>

namespace WebCore {

class CSSSelector;
class ContainerNode;
class Document;
class Element;

namespace Style {
struct SelectorMatchingState;
};

class SelectorDataList {
public:
    explicit SelectorDataList(const CSSSelectorList&);
    bool matches(Element&) const;
    Element* closest(Element&) const;
    Ref<NodeList> queryAll(ContainerNode& rootNode) const;
    Element* queryFirst(ContainerNode& rootNode) const;

    bool shouldStoreInDocument() const { return m_matchType == MatchType::TagNameMatch || m_matchType == MatchType::ClassNameMatch; }
    AtomString classNameToMatch() const;

private:
    struct SelectorData {
        const CSSSelector* selector;
#if ENABLE(CSS_SELECTOR_JIT)
        mutable CompiledSelector compiledSelector { };
#endif
    };

    bool selectorMatches(const SelectorData&, Element&, const ContainerNode& rootNode, Style::SelectorMatchingState* = nullptr) const;
    Element* selectorClosest(const SelectorData&, Element&, const ContainerNode& rootNode, Style::SelectorMatchingState* = nullptr) const;

    template <typename OutputType> void execute(ContainerNode& rootNode, OutputType&) const;
    template <typename OutputType> void executeFastPathForIdSelector(const ContainerNode& rootNode, const SelectorData&, const CSSSelector* idSelector, OutputType&) const;
    template <typename OutputType> void executeSingleTagNameSelectorData(const ContainerNode& rootNode, const SelectorData&, OutputType&) const;
    template <typename OutputType> void executeSingleClassNameSelectorData(const ContainerNode& rootNode, const SelectorData&, OutputType&) const;
    template <typename OutputType> void executeSingleAttributeExactSelectorData(const ContainerNode& rootNode, const SelectorData&, OutputType&) const;
    template <typename OutputType> void executeSingleSelectorData(const ContainerNode& rootNode, const ContainerNode& searchRootNode, const SelectorData&, OutputType&) const;
    template <typename OutputType> void executeSingleMultiSelectorData(const ContainerNode& rootNode, OutputType&) const;
#if ENABLE(CSS_SELECTOR_JIT)
    template <typename Checker, typename OutputType> void executeCompiledSimpleSelectorChecker(const ContainerNode& searchRootNode, Checker, OutputType&, const SelectorData&) const;
    template <typename Checker, typename OutputType> void executeCompiledSelectorCheckerWithCheckingContext(const ContainerNode& rootNode, const ContainerNode& searchRootNode, Checker, OutputType&, const SelectorData&) const;
    template <typename OutputType> void executeCompiledSingleMultiSelectorData(const ContainerNode& rootNode, OutputType&) const;
    static bool compileSelector(const SelectorData&);
#endif // ENABLE(CSS_SELECTOR_JIT)

    Vector<SelectorData> m_selectors;
    mutable enum MatchType {
        CompilableSingle,
        CompilableSingleWithRootFilter,
        CompilableMultipleSelectorMatch,
        CompiledSingle,
        CompiledSingleWithRootFilter,
        CompiledMultipleSelectorMatch,
        SingleSelector,
        SingleSelectorWithRootFilter,
        RightMostWithIdMatch,
        TagNameMatch,
        ClassNameMatch,
        AttributeExactMatch,
        MultipleSelectorMatch,
    } m_matchType;
};

class SelectorQuery {
    WTF_MAKE_TZONE_ALLOCATED(SelectorQuery);
    WTF_MAKE_NONCOPYABLE(SelectorQuery);

public:
    explicit SelectorQuery(CSSSelectorList&&);
    bool matches(Element&) const;
    Element* closest(Element&) const;
    Ref<NodeList> queryAll(ContainerNode& rootNode) const;
    Element* queryFirst(ContainerNode& rootNode) const;

    bool shouldStoreInDocument() const { return m_selectors.shouldStoreInDocument(); }
    AtomString classNameToMatch() const { return m_selectors.classNameToMatch(); }

private:
    CSSSelectorList m_selectorList;
    SelectorDataList m_selectors;
};

class SelectorQueryCache {
    WTF_MAKE_TZONE_ALLOCATED(SelectorQueryCache);
public:
    static SelectorQueryCache& singleton();

    SelectorQuery* add(const String&, const Document&);
    void clear();

private:
    using Key = std::tuple<String, CSSSelectorParserContext, SecurityOriginData>;
    UncheckedKeyHashMap<Key, std::unique_ptr<SelectorQuery>> m_entries;
};

inline bool SelectorQuery::matches(Element& element) const
{
    return m_selectors.matches(element);
}

inline Element* SelectorQuery::closest(Element& element) const
{
    return m_selectors.closest(element);
}

inline Ref<NodeList> SelectorQuery::queryAll(ContainerNode& rootNode) const
{
    return m_selectors.queryAll(rootNode);
}

inline Element* SelectorQuery::queryFirst(ContainerNode& rootNode) const
{
    return m_selectors.queryFirst(rootNode);
}

} // namespace WebCore
