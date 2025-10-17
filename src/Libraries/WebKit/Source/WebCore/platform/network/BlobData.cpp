/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 6, 2023.
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
#include "BlobData.h"

#include "Blob.h"
#include "BlobURL.h"

namespace WebCore {

BlobData::BlobData(const String& contentType)
    : m_contentType(contentType)
{
}

const long long BlobDataItem::toEndOfFile = -1;

long long BlobDataItem::length() const
{
    if (m_length != toEndOfFile)
        return m_length;

    switch (m_type) {
    case Type::Data:
        ASSERT_NOT_REACHED();
        return m_length;
    case Type::File:
        return m_file->size();
    }

    ASSERT_NOT_REACHED();
    return m_length;
}

void BlobData::appendData(Ref<DataSegment>&& data)
{
    auto dataSize = data->size();
    appendData(WTFMove(data), 0, dataSize);
}

void BlobData::appendData(Ref<DataSegment>&& data, long long offset, long long length)
{
    m_items.append(BlobDataItem(WTFMove(data), offset, length));
}

void BlobData::replaceData(const DataSegment& oldData, Ref<DataSegment>&& newData)
{
    for (auto& blobItem : m_items) {
        if (blobItem.data() == &oldData) {
            blobItem.m_data = WTFMove(newData);
            break;
        }
    }
}

void BlobData::appendFile(Ref<BlobDataFileReference>&& file)
{
    file->startTrackingModifications();
    m_items.append(BlobDataItem(WTFMove(file)));
}

Ref<BlobData> BlobData::clone() const
{
    auto blobData = BlobData::create(m_contentType);
    blobData->m_policyContainer = m_policyContainer;
    blobData->m_items = m_items;
    return blobData;
}

void BlobData::appendFile(BlobDataFileReference* file, long long offset, long long length)
{
    m_items.append(BlobDataItem(file, offset, length));
}

} // namespace WebCore
