/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 14, 2024.
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
#include "DeviceIdHashSaltStorage.h"

#include "Logging.h"
#include "PersistencyUtils.h"

#include <WebCore/SharedBuffer.h>
#include <wtf/CryptographicallyRandomNumber.h>
#include <wtf/FileSystem.h>
#include <wtf/HexNumber.h>
#include <wtf/RunLoop.h>
#include <wtf/StdLibExtras.h>
#include <wtf/text/MakeString.h>
#include <wtf/text/StringBuilder.h>
#include <wtf/text/StringHash.h>

namespace WebKit {
using namespace WebCore;

static constexpr unsigned deviceIdHashSaltStorageVersion { 1 };
static constexpr unsigned hashSaltSize { 48 };
static constexpr unsigned randomDataSize { hashSaltSize / 16 };

Ref<DeviceIdHashSaltStorage> DeviceIdHashSaltStorage::create(const String& deviceIdHashSaltStorageDirectory)
{
    auto deviceIdHashSaltStorage = adoptRef(*new DeviceIdHashSaltStorage(deviceIdHashSaltStorageDirectory));
    return deviceIdHashSaltStorage;
}

void DeviceIdHashSaltStorage::completePendingHandler(CompletionHandler<void(HashSet<SecurityOriginData>&&)>&& completionHandler)
{
    ASSERT(RunLoop::isMain());

    HashSet<SecurityOriginData> origins;

    for (auto& hashSaltForOrigin : m_deviceIdHashSaltForOrigins) {
        origins.add(hashSaltForOrigin.value->documentOrigin);
        origins.add(hashSaltForOrigin.value->parentOrigin);
    }

    RunLoop::main().dispatch([origins = WTFMove(origins), completionHandler = WTFMove(completionHandler)]() mutable {
        completionHandler(WTFMove(origins));
    });
}

DeviceIdHashSaltStorage::DeviceIdHashSaltStorage(const String& deviceIdHashSaltStorageDirectory)
    : m_queue(WorkQueue::create("com.apple.WebKit.DeviceIdHashSaltStorage"_s))
    , m_deviceIdHashSaltStorageDirectory(!deviceIdHashSaltStorageDirectory.isEmpty() ? FileSystem::pathByAppendingComponent(deviceIdHashSaltStorageDirectory, String::number(deviceIdHashSaltStorageVersion)) : String())
{
    if (m_deviceIdHashSaltStorageDirectory.isEmpty()) {
        m_isLoaded = true;
        return;
    }

    loadStorageFromDisk([this, protectedThis = Ref { *this }] (auto&& deviceIdHashSaltForOrigins) {
        ASSERT(RunLoop::isMain());
        m_deviceIdHashSaltForOrigins = WTFMove(deviceIdHashSaltForOrigins);
        m_isLoaded = true;

        auto pendingCompletionHandlers = WTFMove(m_pendingCompletionHandlers);
        for (auto& completionHandler : pendingCompletionHandlers)
            completionHandler();
    });
}

DeviceIdHashSaltStorage::~DeviceIdHashSaltStorage()
{
    m_isClosed = true;

    auto pendingCompletionHandlers = WTFMove(m_pendingCompletionHandlers);
    for (auto& completionHandler : pendingCompletionHandlers)
        completionHandler();
}

static std::optional<SecurityOriginData> getSecurityOriginData(ASCIILiteral name, KeyedDecoder* decoder)
{
    String origin;

    if (!decoder->decodeString(name, origin))
        return std::nullopt;

    auto securityOriginData = SecurityOriginData::fromDatabaseIdentifier(origin);
    if (!securityOriginData)
        return std::nullopt;

    return securityOriginData;
}

void DeviceIdHashSaltStorage::loadStorageFromDisk(CompletionHandler<void(HashMap<String, std::unique_ptr<HashSaltForOrigin>>&&)>&& completionHandler)
{
    m_queue->dispatch([this, protectedThis = Ref { *this }, completionHandler = WTFMove(completionHandler)]() mutable {
        ASSERT(!RunLoop::isMain());

        FileSystem::makeAllDirectories(m_deviceIdHashSaltStorageDirectory);

        HashMap<String, std::unique_ptr<HashSaltForOrigin>> deviceIdHashSaltForOrigins;
        for (auto& origin : FileSystem::listDirectory(m_deviceIdHashSaltStorageDirectory)) {
            auto originPath = FileSystem::pathByAppendingComponent(m_deviceIdHashSaltStorageDirectory, origin);
            auto deviceIdHashSalt = URL::fileURLWithFileSystemPath(originPath).lastPathComponent().toString();

            if (hashSaltSize != deviceIdHashSalt.length()) {
                RELEASE_LOG_ERROR(DiskPersistency, "DeviceIdHashSaltStorage: The length of the hash salt (%d) is different to the length of the hash salts defined in WebKit (%d)", deviceIdHashSalt.length(), hashSaltSize);
                continue;
            }

            if (!FileSystem::fileSize(originPath)) {
                RELEASE_LOG_ERROR(DiskPersistency, "DeviceIdHashSaltStorage: Impossible to get the file size of: '%s'", originPath.utf8().data());
                continue;
            }

            auto decoder = createForFile(originPath);

            if (!decoder) {
                RELEASE_LOG_ERROR(DiskPersistency, "DeviceIdHashSaltStorage: Impossible to access the file to restore the hash salt: '%s'", originPath.utf8().data());
                continue;
            }

            auto hashSaltForOrigin = getDataFromDecoder(decoder.get(), WTFMove(deviceIdHashSalt));
            if (!hashSaltForOrigin)
                continue;

            auto origins = makeString(hashSaltForOrigin->documentOrigin.toString(), hashSaltForOrigin->parentOrigin.toString());
            auto deviceIdHashSaltForOrigin = deviceIdHashSaltForOrigins.add(origins, WTFMove(hashSaltForOrigin));

            if (!deviceIdHashSaltForOrigin.isNewEntry)
                RELEASE_LOG_ERROR(DiskPersistency, "DeviceIdHashSaltStorage: There are two files with different hash salts for the same origin: '%s'", originPath.utf8().data());
        }

        RunLoop::main().dispatch([deviceIdHashSaltForOrigins = WTFMove(deviceIdHashSaltForOrigins), completionHandler = WTFMove(completionHandler)]() mutable {
            completionHandler(WTFMove(deviceIdHashSaltForOrigins));
        });
    });
}

std::unique_ptr<DeviceIdHashSaltStorage::HashSaltForOrigin> DeviceIdHashSaltStorage::getDataFromDecoder(KeyedDecoder* decoder, String&& deviceIdHashSalt) const
{
    auto securityOriginData = getSecurityOriginData("origin"_s, decoder);
    if (!securityOriginData) {
        RELEASE_LOG_ERROR(DiskPersistency, "DeviceIdHashSaltStorage: The security origin data in the file is not correct: '%s'", deviceIdHashSalt.utf8().data());
        return nullptr;
    }

    auto parentSecurityOriginData = getSecurityOriginData("parentOrigin"_s, decoder);
    if (!parentSecurityOriginData) {
        RELEASE_LOG_ERROR(DiskPersistency, "DeviceIdHashSaltStorage: The parent security origin data in the file is not correct: '%s'", deviceIdHashSalt.utf8().data());
        return nullptr;
    }

    double lastTimeUsed;
    if (!decoder->decodeDouble("lastTimeUsed"_s, lastTimeUsed)) {
        RELEASE_LOG_ERROR(DiskPersistency, "DeviceIdHashSaltStorage: The last time used was not correctly restored for: '%s'", deviceIdHashSalt.utf8().data());
        return nullptr;
    }

    auto hashSaltForOrigin = makeUnique<HashSaltForOrigin>(WTFMove(securityOriginData.value()), WTFMove(parentSecurityOriginData.value()), WTFMove(deviceIdHashSalt));

    hashSaltForOrigin->lastTimeUsed = WallTime::fromRawSeconds(lastTimeUsed);

    return hashSaltForOrigin;
}

std::unique_ptr<KeyedEncoder> DeviceIdHashSaltStorage::createEncoderFromData(const HashSaltForOrigin& hashSaltForOrigin) const
{
    auto encoder = KeyedEncoder::encoder();
    encoder->encodeString("origin"_s, hashSaltForOrigin.documentOrigin.databaseIdentifier());
    encoder->encodeString("parentOrigin"_s, hashSaltForOrigin.parentOrigin.databaseIdentifier());
    encoder->encodeDouble("lastTimeUsed"_s, hashSaltForOrigin.lastTimeUsed.secondsSinceEpoch().value());
    return encoder;
}

void DeviceIdHashSaltStorage::storeHashSaltToDisk(const HashSaltForOrigin& hashSaltForOrigin)
{
    if (m_deviceIdHashSaltStorageDirectory.isEmpty())
        return;

    m_queue->dispatch([this, protectedThis = Ref { *this }, hashSaltForOrigin = hashSaltForOrigin.isolatedCopy()]() mutable {
        auto encoder = createEncoderFromData(hashSaltForOrigin);
        writeToDisk(WTFMove(encoder), FileSystem::pathByAppendingComponent(m_deviceIdHashSaltStorageDirectory, hashSaltForOrigin.deviceIdHashSalt));
    });
}

void DeviceIdHashSaltStorage::completeDeviceIdHashSaltForOriginCall(SecurityOriginData&& documentOrigin, SecurityOriginData&& parentOrigin, CompletionHandler<void(String&&)>&& completionHandler)
{
    auto origins = makeString(documentOrigin.toString(), parentOrigin.toString());
    auto& deviceIdHashSalt = m_deviceIdHashSaltForOrigins.ensure(origins, [&] {
        std::array<uint64_t, randomDataSize> randomData;
        cryptographicallyRandomValues(asWritableBytes(std::span<uint64_t> { randomData }));

        StringBuilder builder;
        builder.reserveCapacity(hashSaltSize);
        for (uint64_t number : randomData)
            builder.append(hex(number));

        String deviceIdHashSalt = builder.toString();

        auto newHashSaltForOrigin = makeUnique<HashSaltForOrigin>(WTFMove(documentOrigin), WTFMove(parentOrigin), WTFMove(deviceIdHashSalt));

        return newHashSaltForOrigin;
    }).iterator->value;

    deviceIdHashSalt->lastTimeUsed = WallTime::now();

    storeHashSaltToDisk(*deviceIdHashSalt.get());

    completionHandler(String(deviceIdHashSalt->deviceIdHashSalt));
}

void DeviceIdHashSaltStorage::deviceIdHashSaltForOrigin(const SecurityOrigin& documentOrigin, const SecurityOrigin& parentOrigin, CompletionHandler<void(String&&)>&& completionHandler)
{
    ASSERT(RunLoop::isMain());

    if (!m_isLoaded) {
        m_pendingCompletionHandlers.append([this, documentOrigin = documentOrigin.data().isolatedCopy(), parentOrigin = parentOrigin.data().isolatedCopy(), completionHandler = WTFMove(completionHandler)]() mutable {
            completeDeviceIdHashSaltForOriginCall(WTFMove(documentOrigin), WTFMove(parentOrigin), WTFMove(completionHandler));
        });
        return;
    }

    completeDeviceIdHashSaltForOriginCall(SecurityOriginData(documentOrigin.data()), SecurityOriginData(parentOrigin.data()), WTFMove(completionHandler));
}

void DeviceIdHashSaltStorage::getDeviceIdHashSaltOrigins(CompletionHandler<void(HashSet<SecurityOriginData>&&)>&& completionHandler)
{
    ASSERT(RunLoop::isMain());

    if (!m_isLoaded) {
        m_pendingCompletionHandlers.append([this, completionHandler = WTFMove(completionHandler)]() mutable {
            completePendingHandler(WTFMove(completionHandler));
        });
        return;
    }

    completePendingHandler(WTFMove(completionHandler));
}

void DeviceIdHashSaltStorage::deleteHashSaltFromDisk(const HashSaltForOrigin& hashSaltForOrigin)
{
    m_queue->dispatch([this, protectedThis = Ref { *this }, deviceIdHashSalt = hashSaltForOrigin.deviceIdHashSalt.isolatedCopy()]() mutable {
        ASSERT(!RunLoop::isMain());

        String fileFullPath = FileSystem::pathByAppendingComponent(m_deviceIdHashSaltStorageDirectory, deviceIdHashSalt);
        FileSystem::deleteFile(fileFullPath);
    });
}

void DeviceIdHashSaltStorage::deleteDeviceIdHashSaltForOrigins(const Vector<SecurityOriginData>& origins, CompletionHandler<void()>&& completionHandler)
{
    ASSERT(RunLoop::isMain());

    if (!m_isLoaded) {
        m_pendingCompletionHandlers.append([this, origins, completionHandler = WTFMove(completionHandler)]() mutable {
            if (m_isClosed)
                return completionHandler();

            deleteDeviceIdHashSaltForOrigins(origins, WTFMove(completionHandler));
        });
        return;
    }

    m_deviceIdHashSaltForOrigins.removeIf([this, &origins](auto& keyAndValue) {
        bool needsRemoval = origins.contains(keyAndValue.value->documentOrigin) || origins.contains(keyAndValue.value->parentOrigin);
        if (m_deviceIdHashSaltStorageDirectory.isEmpty())
            return needsRemoval;
        if (needsRemoval)
            this->deleteHashSaltFromDisk(*keyAndValue.value.get());
        return needsRemoval;
    });

    RunLoop::main().dispatch(WTFMove(completionHandler));
}

void DeviceIdHashSaltStorage::deleteDeviceIdHashSaltOriginsModifiedSince(WallTime time, CompletionHandler<void()>&& completionHandler)
{
    ASSERT(RunLoop::isMain());

    if (!m_isLoaded) {
        m_pendingCompletionHandlers.append([this, time, completionHandler = WTFMove(completionHandler)]() mutable {
            if (m_isClosed)
                return completionHandler();

            deleteDeviceIdHashSaltOriginsModifiedSince(time, WTFMove(completionHandler));
        });
        return;
    }

    m_deviceIdHashSaltForOrigins.removeIf([this, time](auto& keyAndValue) {
        bool needsRemoval = keyAndValue.value->lastTimeUsed > time;
        if (m_deviceIdHashSaltStorageDirectory.isEmpty())
            return needsRemoval;
        if (needsRemoval)
            this->deleteHashSaltFromDisk(*keyAndValue.value.get());
        return needsRemoval;
    });

    RunLoop::main().dispatch(WTFMove(completionHandler));
}

} // namespace WebKit
