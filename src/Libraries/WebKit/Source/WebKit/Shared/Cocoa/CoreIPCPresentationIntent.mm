/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 28, 2024.
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
#import "config.h"
#import "CoreIPCPresentationIntent.h"

#if PLATFORM(COCOA)

#import <Foundation/Foundation.h>
#import <wtf/RetainPtr.h>
#import <wtf/TZoneMallocInlines.h>

namespace WebKit {

WTF_MAKE_TZONE_ALLOCATED_IMPL(CoreIPCPresentationIntent);

CoreIPCPresentationIntent::CoreIPCPresentationIntent(NSPresentationIntent *intent)
    : m_intentKind(intent.intentKind)
    , m_identity(intent.identity)
{
    if (intent.parentIntent)
        m_parentIntent = makeUnique<CoreIPCPresentationIntent>(intent.parentIntent);

    switch (m_intentKind) {
    case NSPresentationIntentKindHeader:
        m_headerLevel = intent.headerLevel;
        break;
    case NSPresentationIntentKindListItem:
        m_ordinal = intent.ordinal;
        break;
    case NSPresentationIntentKindCodeBlock:
        m_languageHint = { intent.languageHint };
        break;
    case NSPresentationIntentKindTable:
        for (NSNumber *alignment in intent.columnAlignments)
            m_columnAlignments.append(alignment.unsignedIntegerValue);
        m_columnCount = intent.columnCount;
        break;
    case NSPresentationIntentKindTableRow:
        m_row = intent.row;
        break;
    case NSPresentationIntentKindTableCell:
        m_column = intent.column;
        break;
    case NSPresentationIntentKindParagraph:
    case NSPresentationIntentKindOrderedList:
    case NSPresentationIntentKindUnorderedList:
    case NSPresentationIntentKindBlockQuote:
    case NSPresentationIntentKindThematicBreak:
    case NSPresentationIntentKindTableHeaderRow:
        break;
    }
}

RetainPtr<id> CoreIPCPresentationIntent::toID() const
{
    auto parent = (m_parentIntent ? m_parentIntent->toID(): nullptr);
    switch (m_intentKind) {
    case NSPresentationIntentKindParagraph:
        return [NSPresentationIntent paragraphIntentWithIdentity:m_identity nestedInsideIntent:parent.get()];
    case NSPresentationIntentKindHeader:
        return [NSPresentationIntent headerIntentWithIdentity:m_identity level:m_headerLevel nestedInsideIntent:parent.get()];
    case NSPresentationIntentKindOrderedList:
        return [NSPresentationIntent orderedListIntentWithIdentity:m_identity nestedInsideIntent:parent.get()];
    case NSPresentationIntentKindUnorderedList:
        return [NSPresentationIntent unorderedListIntentWithIdentity:m_identity nestedInsideIntent:parent.get()];
    case NSPresentationIntentKindListItem:
        return [NSPresentationIntent listItemIntentWithIdentity:m_identity ordinal:m_ordinal nestedInsideIntent:parent.get()];
    case NSPresentationIntentKindCodeBlock:
        return [NSPresentationIntent codeBlockIntentWithIdentity:m_identity languageHint:m_languageHint nestedInsideIntent:parent.get()];
    case NSPresentationIntentKindBlockQuote:
        return [NSPresentationIntent blockQuoteIntentWithIdentity:m_identity nestedInsideIntent:parent.get()];
    case NSPresentationIntentKindThematicBreak:
        return [NSPresentationIntent thematicBreakIntentWithIdentity:m_identity nestedInsideIntent:parent.get()];
    case NSPresentationIntentKindTable: {
        auto columnAlignments = adoptNS([[NSMutableArray alloc] initWithCapacity:m_columnAlignments.size()]);
        for (int64_t alignment : m_columnAlignments)
            [columnAlignments.get() addObject:@(alignment)];
        return [NSPresentationIntent tableIntentWithIdentity:m_identity columnCount:m_columnCount alignments:columnAlignments.get() nestedInsideIntent:parent.get()];
    }
    case NSPresentationIntentKindTableHeaderRow:
        return [NSPresentationIntent tableHeaderRowIntentWithIdentity:m_identity nestedInsideIntent:parent.get()];
    case NSPresentationIntentKindTableRow:
        return [NSPresentationIntent tableRowIntentWithIdentity:m_identity row:m_row nestedInsideIntent:parent.get()];
    case NSPresentationIntentKindTableCell:
        return [NSPresentationIntent tableCellIntentWithIdentity:m_identity column:m_column nestedInsideIntent:parent.get()];
    }

    ASSERT_NOT_REACHED();
    return nullptr;
}

} // namespace WebKit

#endif // PLATFORM(COCOA)
