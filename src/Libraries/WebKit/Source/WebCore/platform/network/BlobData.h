/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 23, 2022.
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
#ifndef BlobData_h
#define BlobData_h

#include "BlobDataFileReference.h"
#include "PolicyContainer.h"
#include "SharedBuffer.h"
#include <wtf/Forward.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class BlobDataItem {
public:
    WEBCORE_EXPORT static const long long toEndOfFile;

    enum class Type {
        Data,
        File
    };

    Type type() const { return m_type; }

    // For Data type.
    DataSegment* data() const { return m_data.get(); }

    // For File type.
    BlobDataFileReference* file() const { return m_file.get(); }

    long long offset() const { return m_offset; }
    WEBCORE_EXPORT long long length() const; // Computes file length if it's not known yet.

private:
    friend class BlobData;

    explicit BlobDataItem(Ref<BlobDataFileReference>&& file)
        : m_type(Type::File)
        , m_file(WTFMove(file))
        , m_offset(0)
        , m_length(toEndOfFile)
    {
    }

    BlobDataItem(Ref<DataSegment>&& data, long long offset, long long length)
        : m_type(Type::Data)
        , m_data(WTFMove(data))
        , m_offset(offset)
        , m_length(length)
    {
    }

    BlobDataItem(BlobDataFileReference* file, long long offset, long long length)
        : m_type(Type::File)
        , m_file(file)
        , m_offset(offset)
        , m_length(length)
    {
    }

    Type m_type;
    RefPtr<DataSegment> m_data;
    RefPtr<BlobDataFileReference> m_file;

    long long m_offset;
    long long m_length;
};

typedef Vector<BlobDataItem> BlobDataItemList;

class BlobData : public ThreadSafeRefCounted<BlobData> {
public:
    static Ref<BlobData> create(const String& contentType)
    {
        return adoptRef(*new BlobData(contentType));
    }

    const String& contentType() const { return m_contentType; }

    const PolicyContainer& policyContainer() const { return m_policyContainer; }
    void setPolicyContainer(const PolicyContainer& policyContainer) { m_policyContainer = policyContainer; }

    const BlobDataItemList& items() const { return m_items; }

    void replaceData(const DataSegment& oldData, Ref<DataSegment>&& newData);
    void appendData(Ref<DataSegment>&&);
    void appendFile(Ref<BlobDataFileReference>&&);

    Ref<BlobData> clone() const;

private:
    friend class BlobRegistryImpl;
    BlobData(const String& contentType);

    void appendData(Ref<DataSegment>&&, long long offset, long long length);
    void appendFile(BlobDataFileReference*, long long offset, long long length);

    String m_contentType;
    PolicyContainer m_policyContainer;
    BlobDataItemList m_items;
};

} // namespace WebCore

#endif // BlobData_h
