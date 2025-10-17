/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, June 24, 2024.
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

#include "AbstractRange.h"
#include "SimpleRange.h"

namespace JSC {

class AbstractSlotVisitor;

}

namespace WebCore {

template<typename> class ExceptionOr;

class StaticRange final : public AbstractRange, public SimpleRange {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(StaticRange);
public:
    struct Init {
        RefPtr<Node> startContainer;
        unsigned startOffset { 0 };
        RefPtr<Node> endContainer;
        unsigned endOffset { 0 };
    };

    static ExceptionOr<Ref<StaticRange>> create(Init&&);
    WEBCORE_EXPORT static Ref<StaticRange> create(const SimpleRange&);
    static Ref<StaticRange> create(SimpleRange&&);

    Node& startContainer() const final { return SimpleRange::startContainer(); }
    unsigned startOffset() const final { return SimpleRange::startOffset(); }
    Node& endContainer() const final { return SimpleRange::endContainer(); }
    unsigned endOffset() const final { return SimpleRange::endOffset(); }
    bool collapsed() const final { return SimpleRange::collapsed(); }

    // https://dom.spec.whatwg.org/#staticrange-valid
    bool computeValidity() const;

    void visitNodesConcurrently(JSC::AbstractSlotVisitor&) const;

private:
    StaticRange(SimpleRange&&);

    bool isLiveRange() const final { return false; }
};

}

SPECIALIZE_TYPE_TRAITS_BEGIN(WebCore::StaticRange)
    static bool isType(const WebCore::AbstractRange& range) { return !range.isLiveRange(); }
SPECIALIZE_TYPE_TRAITS_END()
