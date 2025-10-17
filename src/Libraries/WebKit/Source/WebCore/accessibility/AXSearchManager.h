/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 21, 2024.
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

#include "AXCoreObject.h"
#include <wtf/HashMap.h>
#include <wtf/Vector.h>

namespace WebCore {

DECLARE_ALLOCATOR_WITH_HEAP_IDENTIFIER(AXSearchManager);

enum class AccessibilitySearchKey {
    AnyType = 1,
    Article,
    BlockquoteSameLevel,
    Blockquote,
    BoldFont,
    Button,
    Checkbox,
    Control,
    DifferentType,
    FontChange,
    FontColorChange,
    Frame,
    Graphic,
    HeadingLevel1,
    HeadingLevel2,
    HeadingLevel3,
    HeadingLevel4,
    HeadingLevel5,
    HeadingLevel6,
    HeadingSameLevel,
    Heading,
    Highlighted,
    ItalicFont,
    KeyboardFocusable,
    Landmark,
    Link,
    List,
    LiveRegion,
    MisspelledWord,
    Outline,
    PlainText,
    RadioGroup,
    SameType,
    StaticText,
    StyleChange,
    TableSameLevel,
    Table,
    TextField,
    Underline,
    UnvisitedLink,
    VisitedLink,
};

struct AccessibilitySearchCriteria {
    // FIXME: change the object pointers to object IDs.
    AXCoreObject* anchorObject { nullptr };
    AXCoreObject* startObject { nullptr };
    CharacterRange startRange;
    AccessibilitySearchDirection searchDirection { AccessibilitySearchDirection::Next };
    Vector<AccessibilitySearchKey> searchKeys;
    String searchText;
    unsigned resultsLimit { 0 };
    bool visibleOnly { false };
    bool immediateDescendantsOnly { false };
};

class AXSearchManager {
    WTF_MAKE_FAST_ALLOCATED_WITH_HEAP_IDENTIFIER(AXSearchManager);
public:
    AXCoreObject::AccessibilityChildrenVector findMatchingObjects(AccessibilitySearchCriteria&&);
    std::optional<AXTextMarkerRange> findMatchingRange(AccessibilitySearchCriteria&&);
private:
    AXCoreObject::AccessibilityChildrenVector findMatchingObjectsInternal(const AccessibilitySearchCriteria&);
    bool matchWithResultsLimit(Ref<AXCoreObject>, const AccessibilitySearchCriteria&, AXCoreObject::AccessibilityChildrenVector&);
    bool match(Ref<AXCoreObject>, const AccessibilitySearchCriteria&);
    bool matchText(Ref<AXCoreObject>, const String&);
    bool matchForSearchKeyAtIndex(Ref<AXCoreObject>, const AccessibilitySearchCriteria&, size_t);

    // Keeps the ranges of misspellings for each object.
    UncheckedKeyHashMap<AXID, Vector<AXTextMarkerRange>> m_misspellingRanges;
};

inline AXCoreObject::AccessibilityChildrenVector AXSearchManager::findMatchingObjects(AccessibilitySearchCriteria&& criteria)
{
    return findMatchingObjectsInternal(std::forward<AccessibilitySearchCriteria>(criteria));
}

WTF::TextStream& operator<<(WTF::TextStream&, AccessibilitySearchKey);
WTF::TextStream& operator<<(WTF::TextStream&, const AccessibilitySearchCriteria&);

} // namespace WebCore
