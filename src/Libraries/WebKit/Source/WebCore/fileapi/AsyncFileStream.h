/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 21, 2021.
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

#include <wtf/Forward.h>
#include <wtf/Function.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/WallTime.h>

namespace WebCore {

class FileStreamClient;
class FileStream;

class WEBCORE_EXPORT AsyncFileStream {
    WTF_MAKE_TZONE_ALLOCATED_EXPORT(AsyncFileStream, WEBCORE_EXPORT);
public:
    explicit AsyncFileStream(FileStreamClient&);
    ~AsyncFileStream();

    void getSize(const String& path, std::optional<WallTime> expectedModificationTime);
    void openForRead(const String& path, long long offset, long long length);
    void close();
    void read(std::span<uint8_t> buffer);

private:
    void start();
    void perform(Function<Function<void(FileStreamClient&)>(FileStream&)>&&);

    struct Internals;
    std::unique_ptr<Internals> m_internals;
};

} // namespace WebCore
