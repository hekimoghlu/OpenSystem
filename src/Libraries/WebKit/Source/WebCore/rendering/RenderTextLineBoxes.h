/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 15, 2024.
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

#include "LayoutRect.h"
#include "RenderObject.h"

namespace WebCore {

class LegacyInlineTextBox;
class RenderStyle;
class RenderText;
class VisiblePosition;

class RenderTextLineBoxes {
public:
    RenderTextLineBoxes();

    LegacyInlineTextBox* first() const { return m_first; }
    LegacyInlineTextBox* last() const { return m_last; }

    LegacyInlineTextBox* createAndAppendLineBox(RenderText&);

    void extract(LegacyInlineTextBox&);
    void attach(LegacyInlineTextBox&);
    void remove(LegacyInlineTextBox&);

    void removeAllFromParent(RenderText&);
    void deleteAll();

    void dirtyAll();
    bool dirtyForTextChange(RenderText&);

    LegacyInlineTextBox* findNext(int offset, int& position) const;

#if ASSERT_ENABLED
    ~RenderTextLineBoxes();
#endif

#if !ASSERT_WITH_SECURITY_IMPLICATION_DISABLED
    void invalidateParentChildLists();
#endif

private:
    void checkConsistency() const;

    LegacyInlineTextBox* m_first;
    LegacyInlineTextBox* m_last;
};

} // namespace WebCore
