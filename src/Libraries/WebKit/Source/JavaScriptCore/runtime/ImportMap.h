/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 30, 2025.
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

#include <wtf/Expected.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/URL.h>
#include <wtf/URLHash.h>
#include <wtf/Vector.h>
#include <wtf/text/AtomStringHash.h>

namespace JSC {

class SourceCode;

class ImportMap final : public RefCounted<ImportMap> {
public:
    using SpecifierMap = UncheckedKeyHashMap<AtomString, URL>;
    using ScopesMap = UncheckedKeyHashMap<URL, SpecifierMap>;
    using ScopesVector = Vector<URL>;
    using IntegrityMap = UncheckedKeyHashMap<URL, String>;

    class Reporter {
    public:
        virtual ~Reporter() = default;
        virtual void reportWarning(const String&) const { };
        virtual void reportError(const String&) const { };
    };

    static Ref<ImportMap> create() { return adoptRef(*new ImportMap()); }

    JS_EXPORT_PRIVATE URL resolve(const String& specifier, const URL& baseURL);

    JS_EXPORT_PRIVATE String integrityForURL(const URL&) const;

    // https://html.spec.whatwg.org/C#parse-an-import-map-string
    JS_EXPORT_PRIVATE static std::optional<Ref<ImportMap>> parseImportMapString(const SourceCode&, const URL& baseURL, const ImportMap::Reporter&);

    // https://html.spec.whatwg.org/C/#merge-existing-and-new-import-maps
    // `newImportMap` is modified in place here, and should not be used after
    // this call.
    JS_EXPORT_PRIVATE void mergeExistingAndNewImportMaps(Ref<ImportMap>&& newImportMap, const ImportMap::Reporter&);

    void addModuleToResolvedModuleSet(String referringScriptURL, AtomString specifier);
private:
    ImportMap() = default;
    ImportMap(SpecifierMap&&, ScopesMap&&, IntegrityMap&&);

    static Expected<URL, String> resolveImportMatch(const AtomString&, const URL&, const SpecifierMap&);

    SpecifierMap m_imports;
    ScopesMap m_scopesMap;
    ScopesVector m_scopesVector;
    IntegrityMap m_integrity;

    // https://html.spec.whatwg.org/C#resolved-module-set
    //
    // We replace the spec's set with two different data structures: a set of all
    // the prefixes resolved at the top-level scope, and a map of scopes to sets
    // of prefixes resolved in them. That permits us to reduce the cost of merging
    // a new map, by performing more work at addModuleToResolvedModuleSet time,
    // and by keeping more prefixes in memory.
    UncheckedKeyHashSet<AtomString> m_toplevelResolvedModuleSet;
    UncheckedKeyHashMap<AtomString, UncheckedKeyHashSet<AtomString>> m_scopedResolvedModuleMap;
};

} // namespace JSC
