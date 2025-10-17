/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 9, 2023.
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
#include <unicode/ubrk.h>
#include <wtf/unicode/icu/ICUHelpers.h>

namespace JSC {

using UBreakIteratorDeleter = ICUDeleter<ubrk_close>;

class IntlSegmenter final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    static void destroy(JSCell* cell)
    {
        static_cast<IntlSegmenter*>(cell)->IntlSegmenter::~IntlSegmenter();
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.intlSegmenterSpace<mode>();
    }

    enum class Granularity : uint8_t { Grapheme, Word, Sentence };

    static IntlSegmenter* create(VM&, Structure*);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    void initializeSegmenter(JSGlobalObject*, JSValue localesValue, JSValue optionsValue);

    JSValue segment(JSGlobalObject*, JSValue) const;
    JSObject* resolvedOptions(JSGlobalObject*) const;

    static JSObject* createSegmentDataObject(JSGlobalObject*, JSString*, int32_t startIndex, int32_t endIndex, UBreakIterator&, Granularity);

private:
    IntlSegmenter(VM&, Structure*);
    DECLARE_DEFAULT_FINISH_CREATION;

    static ASCIILiteral granularityString(Granularity);

    std::unique_ptr<UBreakIterator, UBreakIteratorDeleter> m_segmenter;
    String m_locale;
    Granularity m_granularity { Granularity::Grapheme };
};

} // namespace JSC
