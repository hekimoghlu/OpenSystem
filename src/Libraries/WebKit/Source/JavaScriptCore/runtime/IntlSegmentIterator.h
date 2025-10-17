/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 3, 2024.
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

#include "IntlSegmenter.h"

namespace JSC {

class IntlSegmentIterator final : public JSNonFinalObject {
public:
    using Base = JSNonFinalObject;

    static constexpr DestructionMode needsDestruction = NeedsDestruction;

    static void destroy(JSCell* cell)
    {
        static_cast<IntlSegmentIterator*>(cell)->IntlSegmentIterator::~IntlSegmentIterator();
    }

    template<typename CellType, SubspaceAccess mode>
    static GCClient::IsoSubspace* subspaceFor(VM& vm)
    {
        return vm.intlSegmentIteratorSpace<mode>();
    }

    static IntlSegmentIterator* create(VM&, Structure*, std::unique_ptr<UBreakIterator, UBreakIteratorDeleter>&&, Box<Vector<UChar>>, JSString*, IntlSegmenter::Granularity);
    static Structure* createStructure(VM&, JSGlobalObject*, JSValue);

    DECLARE_INFO;

    JSObject* next(JSGlobalObject*);

    DECLARE_VISIT_CHILDREN;

private:
    IntlSegmentIterator(VM&, Structure*, std::unique_ptr<UBreakIterator, UBreakIteratorDeleter>&&, Box<Vector<UChar>>&&, IntlSegmenter::Granularity, JSString*);

    DECLARE_DEFAULT_FINISH_CREATION;

    std::unique_ptr<UBreakIterator, UBreakIteratorDeleter> m_segmenter;
    Box<Vector<UChar>> m_buffer;
    WriteBarrier<JSString> m_string;
    IntlSegmenter::Granularity m_granularity;
};

} // namespace JSC
