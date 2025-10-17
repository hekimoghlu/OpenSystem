/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 3, 2022.
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
#include "DataTransferItemList.h"

#include "ContextDestructionObserver.h"
#include "DataTransferItem.h"
#include "DeprecatedGlobalSettings.h"
#include "Document.h"
#include "FileList.h"
#include "Pasteboard.h"
#include "Settings.h"
#include <wtf/TZoneMallocInlines.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(DataTransferItemList);

DataTransferItemList::DataTransferItemList(Document& document, DataTransfer& dataTransfer)
    : ContextDestructionObserver(&document)
    , m_dataTransfer(dataTransfer)
{
}

DataTransferItemList::~DataTransferItemList() = default;

unsigned DataTransferItemList::length() const
{
    return ensureItems().size();
}

bool DataTransferItemList::isSupportedPropertyIndex(unsigned index)
{
    return index < ensureItems().size();
}

RefPtr<DataTransferItem> DataTransferItemList::item(unsigned index)
{
    auto& items = ensureItems();
    if (items.size() <= index)
        return nullptr;
    return items[index].copyRef();
}

static bool shouldExposeTypeInItemList(const String& type)
{
    return DeprecatedGlobalSettings::customPasteboardDataEnabled() || Pasteboard::isSafeTypeForDOMToReadAndWrite(type);
}

ExceptionOr<RefPtr<DataTransferItem>> DataTransferItemList::add(Document& document, const String& data, const String& type)
{
    Ref dataTransfer = m_dataTransfer.get();
    if (!dataTransfer->canWriteData())
        return nullptr;

    for (auto& item : ensureItems()) {
        if (!item->isFile() && equalIgnoringASCIICase(item->type(), type))
            return Exception { ExceptionCode::NotSupportedError };
    }

    String lowercasedType = type.convertToASCIILowercase();

    if (!shouldExposeTypeInItemList(lowercasedType))
        return nullptr;

    dataTransfer->setDataFromItemList(document, lowercasedType, data);
    ASSERT(m_items);
    m_items->append(DataTransferItem::create(*this, lowercasedType));
    return m_items->last().ptr();
}

RefPtr<DataTransferItem> DataTransferItemList::add(Ref<File>&& file)
{
    Ref dataTransfer = m_dataTransfer.get();
    if (!dataTransfer->canWriteData())
        return nullptr;

    ensureItems().append(DataTransferItem::create(*this, file->type(), file.copyRef()));
    dataTransfer->didAddFileToItemList();
    return m_items->last().ptr();
}

ExceptionOr<void> DataTransferItemList::remove(unsigned index)
{
    Ref dataTransfer = m_dataTransfer.get();
    if (!dataTransfer->canWriteData())
        return Exception { ExceptionCode::InvalidStateError };

    auto& items = ensureItems();
    if (items.size() <= index)
        return { };

    // FIXME: Remove the file from the pasteboard object once we add support for it.
    Ref removedItem = items[index].copyRef();
    if (!removedItem->isFile())
        dataTransfer->pasteboard().clear(removedItem->type());
    removedItem->clearListAndPutIntoDisabledMode();
    items.remove(index);
    if (removedItem->isFile())
        dataTransfer->updateFileList(protectedScriptExecutionContext().get());

    return { };
}

void DataTransferItemList::clear()
{
    Ref dataTransfer = m_dataTransfer.get();
    dataTransfer->pasteboard().clear();
    bool removedItemContainingFile = false;
    if (m_items) {
        for (auto& item : *m_items) {
            removedItemContainingFile |= item->isFile();
            item->clearListAndPutIntoDisabledMode();
        }
        m_items->clear();
    }

    if (removedItemContainingFile)
        dataTransfer->updateFileList(protectedScriptExecutionContext().get());
}

Vector<Ref<DataTransferItem>>& DataTransferItemList::ensureItems() const
{
    if (m_items)
        return *m_items;

    Ref document = *this->document();
    Ref dataTransfer = m_dataTransfer.get();
    Vector<Ref<DataTransferItem>> items;
    for (auto& type : dataTransfer->typesForItemList(document)) {
        auto lowercasedType = type.convertToASCIILowercase();
        if (shouldExposeTypeInItemList(lowercasedType))
            items.append(DataTransferItem::create(*this, lowercasedType));
    }

    for (auto& file : dataTransfer->files(document).files())
        items.append(DataTransferItem::create(*this, file->type(), file.copyRef()));

    m_items = WTFMove(items);

    return *m_items;
}

static void removeStringItemOfLowercasedType(Vector<Ref<DataTransferItem>>& items, const String& lowercasedType)
{
    auto index = items.findIf([lowercasedType](auto& item) {
        return !item->isFile() && item->type() == lowercasedType;
    });
    if (index == notFound)
        return;
    items[index]->clearListAndPutIntoDisabledMode();
    items.remove(index);
}

void DataTransferItemList::didClearStringData(const String& type)
{
    if (!m_items)
        return;

    auto& items = *m_items;
    if (!type.isNull())
        return removeStringItemOfLowercasedType(items, type.convertToASCIILowercase());

    for (auto& item : items) {
        if (!item->isFile())
            item->clearListAndPutIntoDisabledMode();
    }
    items.removeAllMatching([](auto& item) {
        return !item->isFile();
    });
}

// https://html.spec.whatwg.org/multipage/dnd.html#dom-datatransfer-setdata
void DataTransferItemList::didSetStringData(const String& type)
{
    if (!m_items)
        return;

    String lowercasedType = type.convertToASCIILowercase();
    removeStringItemOfLowercasedType(*m_items, type.convertToASCIILowercase());

    m_items->append(DataTransferItem::create(*this, lowercasedType));
}

Document* DataTransferItemList::document() const
{
    return downcast<Document>(scriptExecutionContext());
}

}

