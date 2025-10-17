/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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
#include <wtf/unicode/icu/ICUHelpers.h>

struct UListFormatter;

namespace JSC {

enum class RelevantExtensionKey : uint8_t;

struct UListFormatterDeleter {
    JS_EXPORT_PRIVATE void operator()(UListFormatter*);
};

class IntlListFormat final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    static void destroy(JSCell* cell)
    {
        static_cast<IntlListFormat*>(cell)->IntlListFormat::~IntlListFormat();
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.intlListFormatSpace<mode>();
    }

    static IntlListFormat* create(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    void initializeListFormat(JSGlobalObject*, JSValue localesValue, JSValue optionsValue);

    JSValue format(JSGlobalObject*, JSValue) const;
    JSValue formatToParts(JSGlobalObject*, JSValue) const;
    JSObject* resolvedOptions(JSGlobalObject*) const;

private:
    IntlListFormat(VM&, Structure*);
    DECLARE_DEFAULT_FINISH_CREATION;

    enum class Type : uint8_t { Conjunction, Disjunction, Unit };
    enum class Style : uint8_t { Short, Long, Narrow };

    static ASCIILiteral typeString(Type);
    static ASCIILiteral styleString(Style);

    std::unique_ptr<UListFormatter, UListFormatterDeleter> m_listFormat;
    String m_locale;
    Type m_type { Type::Conjunction };
    Style m_style { Style::Long };
};

} // namespace JSC
