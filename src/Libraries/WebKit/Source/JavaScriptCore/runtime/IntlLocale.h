/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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

#include "JSObject.h"

namespace JSC {

class IntlLocale final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    static void destroy(JSCell* cell)
    {
        static_cast<IntlLocale*>(cell)->IntlLocale::~IntlLocale();
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.intlLocaleSpace<mode>();
    }

    static IntlLocale* create(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    void initializeLocale(JSGlobalObject*, const String& tag, JSValue optionsValue);
    void initializeLocale(JSGlobalObject*, JSValue tagValue, JSValue optionsValue);
    const String& maximal();
    const String& minimal();
    const String& toString();
    const String& baseName();
    const String& language();
    const String& script();
    const String& region();

    const String& calendar();
    const String& caseFirst();
    const String& collation();
    const String& firstDayOfWeek();
    const String& hourCycle();
    const String& numberingSystem();
    TriState numeric();

    JSArray* calendars(JSGlobalObject*);
    JSArray* collations(JSGlobalObject*);
    JSArray* hourCycles(JSGlobalObject*);
    JSArray* numberingSystems(JSGlobalObject*);
    JSValue timeZones(JSGlobalObject*);
    JSObject* textInfo(JSGlobalObject*);
    JSObject* weekInfo(JSGlobalObject*);

private:
    IntlLocale(VM&, Structure*);
    DECLARE_DEFAULT_FINISH_CREATION;
    DECLARE_VISIT_CHILDREN;

    String keywordValue(ASCIILiteral, bool isBoolean = false) const;

    CString m_localeID;

    String m_maximal;
    String m_minimal;
    String m_fullString;
    String m_baseName;
    String m_language;
    String m_script;
    String m_region;
    std::optional<String> m_calendar;
    std::optional<String> m_caseFirst;
    std::optional<String> m_collation;
    std::optional<String> m_firstDayOfWeek;
    std::optional<String> m_hourCycle;
    std::optional<String> m_numberingSystem;
    TriState m_numeric { TriState::Indeterminate };
};

} // namespace JSC
