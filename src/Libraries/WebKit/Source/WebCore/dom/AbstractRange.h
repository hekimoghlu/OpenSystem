/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 20, 2023.
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

#include <optional>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class Node;

struct SimpleRange;

class AbstractRange : public RefCounted<AbstractRange> {
public:
    virtual ~AbstractRange() = default;

    virtual Node& startContainer() const = 0;
    Ref<Node> protectedStartContainer() const;
    virtual unsigned startOffset() const = 0;
    virtual Node& endContainer() const = 0;
    Ref<Node> protectedEndContainer() const;
    virtual unsigned endOffset() const = 0;
    virtual bool collapsed() const = 0;

    virtual bool isLiveRange() const = 0;
};

WEBCORE_EXPORT SimpleRange makeSimpleRange(const AbstractRange&);
SimpleRange makeSimpleRange(const Ref<AbstractRange>&);
std::optional<SimpleRange> makeSimpleRange(const AbstractRange*);
std::optional<SimpleRange> makeSimpleRange(const RefPtr<AbstractRange>&);

}
