/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 10, 2022.
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
#include <wtf/FilePrintStream.h>

namespace WTF {

FilePrintStream::FilePrintStream(FILE* file, AdoptionMode adoptionMode)
    : m_file(file)
    , m_adoptionMode(adoptionMode)
{
}

FilePrintStream::~FilePrintStream()
{
    if (m_adoptionMode == Borrow)
        return;
    fclose(m_file);
}

std::unique_ptr<FilePrintStream> FilePrintStream::open(const char* filename, const char* mode)
{
    FILE* file = fopen(filename, mode);
    if (!file)
        return nullptr;

    return makeUnique<FilePrintStream>(file);
}

void FilePrintStream::vprintf(const char* format, va_list argList)
{
WTF_ALLOW_UNSAFE_BUFFER_USAGE_BEGIN
    vfprintf(m_file, format, argList);
WTF_ALLOW_UNSAFE_BUFFER_USAGE_END
}

void FilePrintStream::flush()
{
    fflush(m_file);
}

} // namespace WTF

