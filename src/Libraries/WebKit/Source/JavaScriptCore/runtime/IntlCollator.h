/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 17, 2022.
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

#include "IntlObject.h"
#include <unicode/ucol.h>
#include <wtf/unicode/icu/ICUHelpers.h>

namespace JSC {

enum class RelevantExtensionKey : uint8_t;

class JSBoundFunction;

class IntlCollator final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    static void destroy(JSCell* cell)
    {
        static_cast<IntlCollator*>(cell)->IntlCollator::~IntlCollator();
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.intlCollatorSpace<mode>();
    }

    static IntlCollator* create(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    void initializeCollator(JSGlobalObject*, JSValue locales, JSValue optionsValue);
    UCollationResult compareStrings(JSGlobalObject*, StringView, StringView) const;
    JSObject* resolvedOptions(JSGlobalObject*) const;

    JSBoundFunction* boundCompare() const { return m_boundCompare.get(); }
    void setBoundCompare(VM&, JSBoundFunction*);

    bool canDoASCIIUCADUCETComparison() const
    {
        if (m_canDoASCIIUCADUCETComparison == TriState::Indeterminate)
            updateCanDoASCIIUCADUCETComparison();
        return m_canDoASCIIUCADUCETComparison == TriState::True;
    }

#if ASSERT_ENABLED
    static void checkICULocaleInvariants(const LocaleSet&);
#else
    static inline void checkICULocaleInvariants(const LocaleSet&) { }
#endif

private:
    IntlCollator(VM&, Structure*);
    DECLARE_DEFAULT_FINISH_CREATION;
    DECLARE_VISIT_CHILDREN;

    bool updateCanDoASCIIUCADUCETComparison() const;

    static Vector<String> sortLocaleData(const String&, RelevantExtensionKey);
    static Vector<String> searchLocaleData(const String&, RelevantExtensionKey);

    enum class Usage : uint8_t { Sort, Search };
    enum class Sensitivity : uint8_t { Base, Accent, Case, Variant };
    enum class CaseFirst : uint8_t { Upper, Lower, False };

    using UCollatorDeleter = ICUDeleter<ucol_close>;

    static ASCIILiteral usageString(Usage);
    static ASCIILiteral sensitivityString(Sensitivity);
    static ASCIILiteral caseFirstString(CaseFirst);

    WriteBarrier<JSBoundFunction> m_boundCompare;
    std::unique_ptr<UCollator, UCollatorDeleter> m_collator;

    String m_locale;
    String m_collation;
    Usage m_usage;
    Sensitivity m_sensitivity;
    CaseFirst m_caseFirst;
    mutable TriState m_canDoASCIIUCADUCETComparison { TriState::Indeterminate };
    bool m_numeric;
    bool m_ignorePunctuation;
};

} // namespace JSC
