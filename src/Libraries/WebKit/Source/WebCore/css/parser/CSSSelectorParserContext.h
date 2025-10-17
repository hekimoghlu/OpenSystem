/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 21, 2024.
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

#include "CSSParserMode.h"
#include <wtf/HashFunctions.h>
#include <wtf/Hasher.h>

namespace WebCore {

struct CSSParserContext;
class Document;

struct CSSSelectorParserContext {
    CSSParserMode mode { CSSParserMode::HTMLStandardMode };
    bool cssNestingEnabled : 1 { false };
#if ENABLE(SERVICE_CONTROLS)
    bool imageControlsEnabled : 1 { false };
#endif
    bool popoverAttributeEnabled : 1 { false };
    bool targetTextPseudoElementEnabled : 1 { false };
    bool thumbAndTrackPseudoElementsEnabled : 1 { false };
    bool viewTransitionsEnabled : 1 { false };
    bool viewTransitionClassesEnabled : 1 { false };
    bool viewTransitionTypesEnabled : 1 { false };
    bool webkitMediaTextTrackDisplayQuirkEnabled : 1 { false };

    bool isHashTableDeletedValue : 1 { false };

    CSSSelectorParserContext() = default;
    CSSSelectorParserContext(const CSSParserContext&);
    explicit CSSSelectorParserContext(const Document&);

    friend bool operator==(const CSSSelectorParserContext&, const CSSSelectorParserContext&) = default;
};

void add(Hasher&, const CSSSelectorParserContext&);

struct CSSSelectorParserContextHash {
    static unsigned hash(const CSSSelectorParserContext& context) { return computeHash(context); }
    static bool equal(const CSSSelectorParserContext& a, const CSSSelectorParserContext& b) { return a == b; }
    static const bool safeToCompareToEmptyOrDeleted = false;
};

} // namespace WebCore

namespace WTF {

template<> struct HashTraits<WebCore::CSSSelectorParserContext> : GenericHashTraits<WebCore::CSSSelectorParserContext> {
    static void constructDeletedValue(WebCore::CSSSelectorParserContext& slot) { slot.isHashTableDeletedValue = true; }
    static bool isDeletedValue(const WebCore::CSSSelectorParserContext& value) { return value.isHashTableDeletedValue; }
    static WebCore::CSSSelectorParserContext emptyValue() { return { }; }
};

template<> struct DefaultHash<WebCore::CSSSelectorParserContext> : WebCore::CSSSelectorParserContextHash { };

} // namespace WTF
