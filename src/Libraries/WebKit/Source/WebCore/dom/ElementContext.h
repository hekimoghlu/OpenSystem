/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 2, 2022.
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

#include "ElementIdentifier.h"
#include "FloatRect.h"
#include "PageIdentifier.h"
#include "ProcessQualified.h"
#include "ScriptExecutionContextIdentifier.h"
#include <wtf/ObjectIdentifier.h>

namespace WebCore {

struct ElementContext {
    FloatRect boundingRect;

    Markable<PageIdentifier> webPageIdentifier;
    Markable<ScriptExecutionContextIdentifier> documentIdentifier;
    Markable<ElementIdentifier> elementIdentifier;

    ~ElementContext() = default;

    bool isSameElement(const ElementContext& other) const
    {
        return webPageIdentifier == other.webPageIdentifier && documentIdentifier == other.documentIdentifier && elementIdentifier == other.elementIdentifier;
    }
};

inline bool operator==(const ElementContext& a, const ElementContext& b)
{
    return a.boundingRect == b.boundingRect && a.isSameElement(b);
}
}
