/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, January 14, 2022.
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

#include "FormData.h"
#include <wtf/FileSystem.h>
#include <wtf/Vector.h>

namespace WebCore {

class CurlFormDataStream {
public:
    explicit CurlFormDataStream(const RefPtr<FormData>&);
    WEBCORE_EXPORT ~CurlFormDataStream();

    void clean();

    const Vector<uint8_t>* getPostData();
    unsigned long long totalSize();

    std::optional<size_t> read(char*, size_t);
    unsigned long long totalReadSize() { return m_totalReadSize; }

private:
    std::optional<size_t> readFromFile(const FormDataElement::EncodedFileData&, char*, size_t);
    std::optional<size_t> readFromData(const Vector<uint8_t>&, char*, size_t);

    RefPtr<FormData> m_formData;

    std::unique_ptr<Vector<uint8_t>> m_postData;
    bool m_isContentLengthUpdated { false };
    unsigned long long m_totalSize { 0 };
    unsigned long long m_totalReadSize { 0 };

    size_t m_elementPosition { 0 };

    FileSystem::PlatformFileHandle m_fileHandle { FileSystem::invalidPlatformFileHandle };
    size_t m_dataOffset { 0 };
};

} // namespace WebCore
