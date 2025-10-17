/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, December 15, 2022.
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
#include "DataTransferItem.h"

#include "DOMFileSystem.h"
#include "DataTransferItemList.h"
#include "Document.h"
#include "File.h"
#include "FileSystemDirectoryEntry.h"
#include "FileSystemFileEntry.h"
#include "ScriptExecutionContext.h"
#include "StringCallback.h"
#include <wtf/FileSystem.h>

namespace WebCore {

Ref<DataTransferItem> DataTransferItem::create(WeakPtr<DataTransferItemList>&& list, const String& type)
{
    return adoptRef(*new DataTransferItem(WTFMove(list), type));
}

Ref<DataTransferItem> DataTransferItem::create(WeakPtr<DataTransferItemList>&& list, const String& type, Ref<File>&& file)
{
    return adoptRef(*new DataTransferItem(WTFMove(list), type, WTFMove(file)));
}

DataTransferItem::DataTransferItem(WeakPtr<DataTransferItemList>&& list, const String& type)
    : m_list(WTFMove(list))
    , m_type(type)
{
}

DataTransferItem::DataTransferItem(WeakPtr<DataTransferItemList>&& list, const String& type, Ref<File>&& file)
    : m_list(WTFMove(list))
    , m_type(type)
    , m_file(WTFMove(file))
{
}

DataTransferItem::~DataTransferItem() = default;

void DataTransferItem::clearListAndPutIntoDisabledMode()
{
    m_list.clear();
}

String DataTransferItem::kind() const
{
    return m_file ? "file"_s : "string"_s;
}

String DataTransferItem::type() const
{
    return isInDisabledMode() ? String() : m_type;
}

void DataTransferItem::getAsString(Document& document, RefPtr<StringCallback>&& callback) const
{
    if (!callback || !m_list || m_file)
        return;

    Ref dataTransfer = m_list->dataTransfer();
    if (!dataTransfer->canReadData())
        return;

    // FIXME: Make this async.
    callback->scheduleCallback(document, dataTransfer->getDataForItem(document, m_type));
}

RefPtr<File> DataTransferItem::getAsFile() const
{
    if (!m_list || !m_list->dataTransfer().canReadData())
        return nullptr;
    return m_file.copyRef();
}

RefPtr<FileSystemEntry> DataTransferItem::getAsEntry(ScriptExecutionContext& context) const
{
    auto file = getAsFile();
    if (!file)
        return nullptr;

    return DOMFileSystem::createEntryForFile(context, *file);
}

} // namespace WebCore
