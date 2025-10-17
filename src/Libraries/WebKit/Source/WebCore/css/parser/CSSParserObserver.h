/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 16, 2022.
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

#include "StyleRuleType.h"

namespace WebCore {

class CSSParserToken;
class CSSParserTokenRange;

// This is only for the inspector and shouldn't be used elsewhere.
class CSSParserObserver {
public:
    virtual ~CSSParserObserver() { };
    virtual void startRuleHeader(StyleRuleType, unsigned offset) = 0;
    virtual void endRuleHeader(unsigned offset) = 0;
    virtual void observeSelector(unsigned startOffset, unsigned endOffset) = 0;
    virtual void startRuleBody(unsigned offset) = 0;
    virtual void endRuleBody(unsigned offset) = 0;
    virtual void markRuleBodyContainsImplicitlyNestedProperties() = 0;
    virtual void observeProperty(unsigned startOffset, unsigned endOffset, bool isImportant, bool isParsed) = 0;
    virtual void observeComment(unsigned startOffset, unsigned endOffset) = 0;
};

} // namespace WebCore
