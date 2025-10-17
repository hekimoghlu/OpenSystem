/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, November 6, 2023.
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
#include "config.h"
#include "UndoManager.h"

#include "CustomUndoStep.h"
#include "Document.h"
#include "Editor.h"
#include "FrameDestructionObserverInlines.h"
#include "LocalFrame.h"
#include "UndoItem.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(UndoManager);

UndoManager::UndoManager(Document& document)
    : m_document(document)
{
}

UndoManager::~UndoManager() = default;

ExceptionOr<void> UndoManager::addItem(Ref<UndoItem>&& item)
{
    if (item->undoManager())
        return Exception { ExceptionCode::InvalidModificationError, "This item has already been added to an UndoManager"_s };

    RefPtr frame = m_document.frame();
    if (!frame)
        return Exception { ExceptionCode::SecurityError, "A browsing context is required to add an UndoItem"_s };

    item->setUndoManager(this);
    frame->protectedEditor()->registerCustomUndoStep(CustomUndoStep::create(item));
    m_items.add(WTFMove(item));
    return { };
}

void UndoManager::removeItem(UndoItem& item)
{
    if (auto foundItem = m_items.take(&item))
        foundItem->setUndoManager(nullptr);
}

void UndoManager::removeAllItems()
{
    for (auto& item : m_items)
        item->setUndoManager(nullptr);
    m_items.clear();
}

} // namespace WebCore
