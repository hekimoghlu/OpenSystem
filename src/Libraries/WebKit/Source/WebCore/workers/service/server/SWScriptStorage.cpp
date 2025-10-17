/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 11, 2023.
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
#include "SWScriptStorage.h"

#include "Logging.h"
#include "ScriptBuffer.h"
#include "ServiceWorkerRegistrationKey.h"
#include <pal/crypto/CryptoDigest.h>
#include <wtf/MainThread.h>
#include <wtf/PageBlock.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/Base64.h>

namespace WebCore {

WTF_MAKE_TZONE_ALLOCATED_IMPL(SWScriptStorage);

static bool shouldUseFileMapping(uint64_t fileSize)
{
    return fileSize >= pageSize();
}

SWScriptStorage::SWScriptStorage(const String& directory)
    : m_directory(directory)
    , m_salt(valueOrDefault(FileSystem::readOrMakeSalt(saltPath())))
{
    ASSERT(!isMainThread());
}

String SWScriptStorage::sha2Hash(const String& input) const
{
    auto crypto = PAL::CryptoDigest::create(PAL::CryptoDigest::Algorithm::SHA_256);
    crypto->addBytes(m_salt);
    auto inputUTF8 = input.utf8();
    crypto->addBytes(inputUTF8.span());
    return base64URLEncodeToString(crypto->computeHash());
}

String SWScriptStorage::sha2Hash(const URL& input) const
{
    return sha2Hash(input.string());
}

String SWScriptStorage::saltPath() const
{
    return FileSystem::pathByAppendingComponent(m_directory, "salt"_s);
}

String SWScriptStorage::registrationDirectory(const ServiceWorkerRegistrationKey& registrationKey) const
{
    return FileSystem::pathByAppendingComponents(m_directory, { sha2Hash(registrationKey.topOrigin().toString()), sha2Hash(registrationKey.scope()) });
}

String SWScriptStorage::scriptPath(const ServiceWorkerRegistrationKey& registrationKey, const URL& scriptURL) const
{
    return FileSystem::pathByAppendingComponent(registrationDirectory(registrationKey), sha2Hash(scriptURL));
}

ScriptBuffer SWScriptStorage::store(const ServiceWorkerRegistrationKey& registrationKey, const URL& scriptURL, const ScriptBuffer& script)
{
    ASSERT(!isMainThread());

    auto scriptPath = this->scriptPath(registrationKey, scriptURL);
    FileSystem::makeAllDirectories(FileSystem::parentPath(scriptPath));

    size_t size = script.buffer() ? script.buffer()->size() : 0;

    auto iterateOverBufferAndWriteData = [&](const Function<bool(std::span<const uint8_t>)>& writeData) {
        script.buffer()->forEachSegment([&](std::span<const uint8_t> span) {
            writeData(span);
        });
    };

    // Make sure we delete the file before writing as there may be code using a mmap'd version of this file.
    FileSystem::deleteFile(scriptPath);

    if (!shouldUseFileMapping(size)) {
        auto handle = FileSystem::openFile(scriptPath, FileSystem::FileOpenMode::Truncate);
        if (!FileSystem::isHandleValid(handle)) {
            RELEASE_LOG_ERROR(ServiceWorker, "SWScriptStorage::store: Failure to store %s, FileSystem::openFile() failed", scriptPath.utf8().data());
            return { };
        }
        if (size) {
            iterateOverBufferAndWriteData([&](std::span<const uint8_t> span) {
                FileSystem::writeToFile(handle, span);
                return true;
            });
        }
        FileSystem::closeFile(handle);
        return script;
    }

    auto mappedFile = FileSystem::mapToFile(scriptPath, size, WTFMove(iterateOverBufferAndWriteData));
    if (!mappedFile) {
        RELEASE_LOG_ERROR(ServiceWorker, "SWScriptStorage::store: Failure to store %s, FileSystem::mapToFile() failed", scriptPath.utf8().data());
        return { };
    }
    return ScriptBuffer { SharedBuffer::create(WTFMove(mappedFile)) };
}

ScriptBuffer SWScriptStorage::retrieve(const ServiceWorkerRegistrationKey& registrationKey, const URL& scriptURL)
{
    ASSERT(!isMainThread());

    auto scriptPath = this->scriptPath(registrationKey, scriptURL);
    auto fileSize = FileSystem::fileSize(scriptPath);
    if (!fileSize) {
        RELEASE_LOG_ERROR(ServiceWorker, "SWScriptStorage::retrieve: Failure to retrieve %s, FileSystem::fileSize() failed", scriptPath.utf8().data());
        return { };
    }

    // FIXME: Do we need to disable file mapping in more cases to avoid having too many file descriptors open?
    RefPtr<FragmentedSharedBuffer> buffer = SharedBuffer::createWithContentsOfFile(scriptPath, FileSystem::MappedFileMode::Private, shouldUseFileMapping(*fileSize) ? SharedBuffer::MayUseFileMapping::Yes : SharedBuffer::MayUseFileMapping::No);
    return buffer;
}

void SWScriptStorage::clear(const ServiceWorkerRegistrationKey& registrationKey)
{
    auto registrationDirectory = this->registrationDirectory(registrationKey);
    bool result = FileSystem::deleteNonEmptyDirectory(registrationDirectory);
    RELEASE_LOG_ERROR_IF(!result, ServiceWorker, "SWScriptStorage::clear: Failure to clear scripts for registration %s", registrationKey.toDatabaseKey().utf8().data());
}

} // namespace WebCore
