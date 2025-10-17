/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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

#include "ActiveDOMObject.h"
#include "CSSFontFace.h"
#include "CSSFontFaceSet.h"
#include "CachedResourceHandle.h"
#include "Font.h"
#include "FontSelector.h"
#include "Timer.h"
#include "WebKitFontFamilyNames.h"
#include <memory>
#include <wtf/Forward.h>
#include <wtf/HashSet.h>
#include <wtf/RefPtr.h>
#include <wtf/text/StringHash.h>

namespace WebCore {

class CSSPrimitiveValue;
class CSSSegmentedFontFace;
class CSSValueList;
class CachedFont;
class ScriptExecutionContext;
class StyleRuleFontFace;
class StyleRuleFontFeatureValues;
class StyleRuleFontPaletteValues;

class CSSFontSelector final : public FontSelector, public CSSFontFaceClient, public ActiveDOMObject {
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    USING_CAN_MAKE_WEAKPTR(FontSelector);

    using FontSelector::ref;
    using FontSelector::deref;

    static Ref<CSSFontSelector> create(ScriptExecutionContext&);
    virtual ~CSSFontSelector();

    unsigned version() const final { return m_version; }
    unsigned uniqueId() const final { return m_uniqueId; }

    FontRanges fontRangesForFamily(const FontDescription&, const AtomString&) final;
    size_t fallbackFontCount() final;
    RefPtr<Font> fallbackFontAt(const FontDescription&, size_t) final;

    void clearFonts();
    void emptyCaches();
    void buildStarted();
    void buildCompleted();

    void addFontFaceRule(StyleRuleFontFace&, bool isInitiatingElementInUserAgentShadowTree);
    void addFontPaletteValuesRule(const StyleRuleFontPaletteValues&);
    void addFontFeatureValuesRule(const StyleRuleFontFeatureValues&);

    void fontCacheInvalidated() final;

    bool isEmpty() const;

    void registerForInvalidationCallbacks(FontSelectorClient&) final;
    void unregisterForInvalidationCallbacks(FontSelectorClient&) final;

    bool isSimpleFontSelectorForDescription() const final;

    bool isCSSFontSelector() const final { return true; }

    ScriptExecutionContext* scriptExecutionContext() const { return m_context.get(); }

    FontFaceSet* fontFaceSetIfExists();
    FontFaceSet& fontFaceSet();
    CSSFontFaceSet& cssFontFaceSet() { return m_cssFontFaceSet; }

    void incrementIsComputingRootStyleFont() { ++m_computingRootStyleFontCount; }
    void decrementIsComputingRootStyleFont() { --m_computingRootStyleFontCount; }

    void loadPendingFonts();

    void updateStyleIfNeeded();

private:
    explicit CSSFontSelector(ScriptExecutionContext&);

    void dispatchInvalidationCallbacks();

    void opportunisticallyStartFontDataURLLoading(const FontCascadeDescription&, const AtomString& family) final;

    std::optional<AtomString> resolveGenericFamily(const FontDescription&, const AtomString& family);

    const FontPaletteValues& lookupFontPaletteValues(const AtomString& familyName, const FontDescription&);
    RefPtr<FontFeatureValues> lookupFontFeatureValues(const AtomString& familyName);

    // CSSFontFaceClient
    void fontLoaded(CSSFontFace&) final;
    void updateStyleIfNeeded(CSSFontFace&) final;

    void fontModified();

    struct PendingFontFaceRule {
        StyleRuleFontFace& styleRuleFontFace;
        bool isInitiatingElementInUserAgentShadowTree;
    };
    Vector<PendingFontFaceRule> m_stagingArea;

    WeakPtr<ScriptExecutionContext> m_context;
    RefPtr<FontFaceSet> m_fontFaceSet;
    Ref<CSSFontFaceSet> m_cssFontFaceSet;
    UncheckedKeyHashSet<FontSelectorClient*> m_clients;

    struct PaletteMapHash : DefaultHash<std::pair<AtomString, AtomString>> {
        static unsigned hash(const std::pair<AtomString, AtomString>& key)
        {
            return pairIntHash(ASCIICaseInsensitiveHash::hash(key.first), DefaultHash<AtomString>::hash(key.second));
        }

        static bool equal(const std::pair<AtomString, AtomString>& a, const std::pair<AtomString, AtomString>& b)
        {
            return ASCIICaseInsensitiveHash::equal(a.first, b.first) && DefaultHash<AtomString>::equal(a.second, b.second);
        }
    };
    UncheckedKeyHashMap<std::pair<AtomString, AtomString>, FontPaletteValues, PaletteMapHash> m_paletteMap;
    UncheckedKeyHashMap<String, Ref<FontFeatureValues>> m_featureValues;

    UncheckedKeyHashSet<RefPtr<CSSFontFace>> m_cssConnectionsPossiblyToRemove;
    UncheckedKeyHashSet<RefPtr<StyleRuleFontFace>> m_cssConnectionsEncounteredDuringBuild;

    CSSFontFaceSet::FontModifiedObserver m_fontModifiedObserver;

    unsigned m_uniqueId;
    unsigned m_version;
    unsigned m_computingRootStyleFontCount { 0 };
    bool m_creatingFont { false };
    bool m_buildIsUnderway { false };
    bool m_isStopped { false };

    WebKitFontFamilyNames::FamilyNamesList<AtomString> m_fontFamilyNames;
};

} // namespace WebCore

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::CSSFontSelector)
    static bool isType(const WebCore::FontSelector& selector) { return selector.isCSSFontSelector(); }
SPECIALIZE_TYPE_TRAITS_END()
