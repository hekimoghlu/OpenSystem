/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 5, 2024.
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
#import "RevealItem.h"

#import "ArgumentCodersCocoa.h"
#import <pal/cocoa/RevealSoftLink.h>

namespace WebKit {

#if ENABLE(REVEAL)

RevealItemRange::RevealItemRange(NSRange range)
    : location(range.location)
    , length(range.length)
{
}

RevealItem::RevealItem(const String& text, RevealItemRange selectedRange)
    : m_text(text)
    , m_selectedRange(selectedRange)
{
}

NSRange RevealItem::highlightRange() const
{
    return item().highlightRange;
}

RVItem *RevealItem::item() const
{
    if (!m_item)
        m_item = adoptNS([PAL::allocRVItemInstance() initWithText:m_text selectedRange:NSMakeRange(m_selectedRange.location, m_selectedRange.length)]);
    return m_item.get();
}

#endif // ENABLE(REVEAL)

} // namespace WebKit
