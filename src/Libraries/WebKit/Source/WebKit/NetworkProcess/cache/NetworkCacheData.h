/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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

#include <span>
#include <wtf/FileSystem.h>
#include <wtf/SHA1.h>
#include <wtf/ThreadSafeRefCounted.h>
#include <wtf/text/WTFString.h>

#if PLATFORM(COCOA)
#include <wtf/OSObjectPtr.h>
#endif

#if USE(GLIB)
#include <wtf/glib/GRefPtr.h>
#endif

#if USE(CURL)
#include <variant>
#include <wtf/Box.h>
#endif

namespace WebCore {
class SharedMemory;
}

namespace WebKit {

namespace NetworkCache {

class Data {
public:
    Data() { }
    Data(std::span<const uint8_t>);

    ~Data() { }

    static Data empty();
    static Data adoptMap(FileSystem::MappedFileData&&, FileSystem::PlatformFileHandle);

#if PLATFORM(COCOA)
    enum class Backing { Buffer, Map };
    Data(OSObjectPtr<dispatch_data_t>&&, Backing = Backing::Buffer);
#endif
#if USE(GLIB)
    Data(GRefPtr<GBytes>&&, FileSystem::PlatformFileHandle fd = FileSystem::invalidPlatformFileHandle);
#elif USE(CURL)
    Data(std::variant<Vector<uint8_t>, FileSystem::MappedFileData>&&);
    Data(Vector<uint8_t>&& data) : Data(std::variant<Vector<uint8_t>, FileSystem::MappedFileData> { WTFMove(data) }) { }
#endif
    bool isNull() const;
    bool isEmpty() const { return !size(); }

    std::span<const uint8_t> span() const;
    size_t size() const;
    bool isMap() const { return m_isMap; }
    RefPtr<WebCore::SharedMemory> tryCreateSharedMemory() const;

    Data subrange(size_t offset, size_t) const;

    bool apply(const Function<bool(std::span<const uint8_t>)>&) const;

    Data mapToFile(const String& path) const;

#if PLATFORM(COCOA)
    dispatch_data_t dispatchData() const { return m_dispatchData.get(); }
#endif

#if USE(GLIB)
    GBytes* bytes() const { return m_buffer.get(); }
#endif
private:
#if PLATFORM(COCOA)
    mutable OSObjectPtr<dispatch_data_t> m_dispatchData;
    mutable std::span<const uint8_t> m_data;
#endif
#if USE(GLIB)
    mutable GRefPtr<GBytes> m_buffer;
    FileSystem::PlatformFileHandle m_fileDescriptor { FileSystem::invalidPlatformFileHandle };
#endif
#if USE(CURL)
    Box<std::variant<Vector<uint8_t>, FileSystem::MappedFileData>> m_buffer;
#endif
    bool m_isMap { false };
};

Data concatenate(const Data&, const Data&);
bool bytesEqual(const Data&, const Data&);
Data adoptAndMapFile(FileSystem::PlatformFileHandle, size_t offset, size_t);
Data mapFile(const String& path);

using Salt = FileSystem::Salt;
SHA1::Digest computeSHA1(const Data&, const Salt&);

}

}
