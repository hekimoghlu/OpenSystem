/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 19, 2023.
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

#include "CSSFontFace.h"
#include <variant>
#include <wtf/HashMap.h>
#include <wtf/Observer.h>
#include <wtf/Vector.h>
#include <wtf/WeakHashSet.h>
#include <wtf/text/StringHash.h>

namespace WebCore {
struct FontEventClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::FontEventClient> : std::true_type { };
}

namespace WebCore {

class CSSPrimitiveValue;
class CSSSegmentedFontFace;
class FontFaceSet;

template<typename> class ExceptionOr;

struct FontEventClient : public CanMakeWeakPtr<FontEventClient> {
    virtual ~FontEventClient() = default;
    virtual void faceFinished(CSSFontFace&, CSSFontFace::Status) = 0;
    virtual void startedLoading() = 0;
    virtual void completedLoading() = 0;
};

class CSSFontFaceSet final : public RefCounted<CSSFontFaceSet>, public CSSFontFaceClient {
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<CSSFontFaceSet> create(CSSFontSelector* owningFontSelector = nullptr)
    {
        return adoptRef(*new CSSFontFaceSet(owningFontSelector));
    }
    ~CSSFontFaceSet();

    using FontModifiedObserver = Observer<void()>;
    void addFontModifiedObserver(const FontModifiedObserver&);

    void addFontEventClient(const FontEventClient&);

    // Calling updateStyleIfNeeded() might delete |this|.
    void updateStyleIfNeeded();

    bool hasFace(const CSSFontFace&) const;
    size_t faceCount() const { return m_faces.size(); }
    void add(CSSFontFace&);
    void remove(const CSSFontFace&);
    void purge();
    void emptyCaches();
    void clear();
    CSSFontFace& operator[](size_t i);

    CSSFontFace* lookUpByCSSConnection(StyleRuleFontFace&);

    ExceptionOr<bool> check(const String& font, const String& text);

    CSSSegmentedFontFace* fontFace(FontSelectionRequest, const AtomString& family);

    enum class Status { Loading, Loaded };
    Status status() const { return m_status; }

    bool hasActiveFontFaces() { return status() == Status::Loading; }

    size_t facesPartitionIndex() const { return m_facesPartitionIndex; }

    ExceptionOr<Vector<std::reference_wrapper<CSSFontFace>>> matchingFacesExcludingPreinstalledFonts(const String& font, const String& text);

    // FIXME: Should this be implemented?
    void updateStyleIfNeeded(CSSFontFace&) final { }

private:
    CSSFontFaceSet(CSSFontSelector*);

    void removeFromFacesLookupTable(const CSSFontFace&, const CSSValueList& familiesToSearchFor);
    void addToFacesLookupTable(CSSFontFace&);

    void incrementActiveCount();
    void decrementActiveCount();

    void fontStateChanged(CSSFontFace&, CSSFontFace::Status oldState, CSSFontFace::Status newState) final;
    void fontPropertyChanged(CSSFontFace&, CSSValueList* oldFamilies = nullptr) final;

    void ensureLocalFontFacesForFamilyRegistered(const AtomString&);

    static String familyNameFromPrimitive(const CSSPrimitiveValue&);

    using FontSelectionKey = std::optional<FontSelectionRequest>;
    struct FontSelectionKeyHash {
        static unsigned hash(const FontSelectionKey& key) { return computeHash(key); }
        static bool equal(const FontSelectionKey& a, const FontSelectionKey& b) { return a == b; }
        static const bool safeToCompareToEmptyOrDeleted = true;
    };
    struct FontSelectionKeyHashTraits : SimpleClassHashTraits<FontSelectionKey> {
        static const bool emptyValueIsZero = false;
        static FontSelectionKey emptyValue() { return FontSelectionRequest { }; }
        static void constructDeletedValue(FontSelectionKey& slot) { slot = std::nullopt; }
        static bool isDeletedValue(const FontSelectionKey& value) { return !value; }
    };
    using FontSelectionHashMap = UncheckedKeyHashMap<FontSelectionKey, RefPtr<CSSSegmentedFontFace>, FontSelectionKeyHash, FontSelectionKeyHashTraits>;

    // m_faces should hold all the same fonts as the ones inside inside m_facesLookupTable.
    Vector<Ref<CSSFontFace>> m_faces; // We should investigate moving m_faces to FontFaceSet and making it reference FontFaces. This may clean up the font loading design.
    UncheckedKeyHashMap<String, Vector<Ref<CSSFontFace>>, ASCIICaseInsensitiveHash> m_facesLookupTable;
    UncheckedKeyHashMap<String, Vector<Ref<CSSFontFace>>, ASCIICaseInsensitiveHash> m_locallyInstalledFacesLookupTable;
    UncheckedKeyHashMap<String, FontSelectionHashMap, ASCIICaseInsensitiveHash> m_cache;
    UncheckedKeyHashMap<StyleRuleFontFace*, CSSFontFace*> m_constituentCSSConnections;
    size_t m_facesPartitionIndex { 0 }; // All entries in m_faces before this index are CSS-connected.
    Status m_status { Status::Loaded };
    WeakHashSet<FontModifiedObserver> m_fontModifiedObservers;
    WeakHashSet<FontEventClient> m_fontEventClients;
    WeakPtr<CSSFontSelector> m_owningFontSelector;
    unsigned m_activeCount { 0 };
};

}
