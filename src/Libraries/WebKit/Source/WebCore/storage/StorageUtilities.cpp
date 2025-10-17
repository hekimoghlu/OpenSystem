/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 26, 2022.
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
#include "StorageUtilities.h"

#include "ClientOrigin.h"
#include "WebCorePersistentCoders.h"
#include <pal/crypto/CryptoDigest.h>
#include <wtf/FileSystem.h>
#include <wtf/Scope.h>
#include <wtf/persistence/PersistentCoders.h>
#include <wtf/text/Base64.h>

#if ASSERT_ENABLED
#include <wtf/RunLoop.h>
#endif

namespace WebCore {
namespace StorageUtilities {

std::optional<ClientOrigin> readOriginFromFile(const String& filePath)
{
    ASSERT(!RunLoop::isMain());

    if (filePath.isEmpty() || !FileSystem::fileExists(filePath))
        return std::nullopt;

    auto originFileHandle = FileSystem::openFile(filePath, FileSystem::FileOpenMode::Read);
    auto closeFile = makeScopeExit([&] {
        FileSystem::closeFile(originFileHandle);
    });

    if (!FileSystem::isHandleValid(originFileHandle))
        return std::nullopt;

    auto originContent = FileSystem::readEntireFile(originFileHandle);
    if (!originContent)
        return std::nullopt;

    WTF::Persistence::Decoder decoder(originContent->span());
    std::optional<ClientOrigin> origin;
    decoder >> origin;
    return origin;
}

bool writeOriginToFile(const String& filePath, const ClientOrigin& origin)
{
    if (filePath.isEmpty() || FileSystem::fileExists(filePath))
        return false;

    FileSystem::makeAllDirectories(FileSystem::parentPath(filePath));
    auto originFileHandle = FileSystem::openFile(filePath, FileSystem::FileOpenMode::ReadWrite);
    auto closeFile = makeScopeExit([&] {
        FileSystem::closeFile(originFileHandle);
    });

    if (!FileSystem::isHandleValid(originFileHandle)) {
        LOG_ERROR("writeOriginToFile: Failed to open origin file '%s'", filePath.utf8().data());
        return false;
    }

    WTF::Persistence::Encoder encoder;
    encoder << origin;
    FileSystem::writeToFile(originFileHandle, encoder.span());
    return true;
}

String encodeSecurityOriginForFileName(FileSystem::Salt salt, const SecurityOriginData& origin)
{
    auto crypto = PAL::CryptoDigest::create(PAL::CryptoDigest::Algorithm::SHA_256);
    auto originString = origin.toString().utf8();
    crypto->addBytes(originString.span());
    crypto->addBytes(salt);
    return base64URLEncodeToString(crypto->computeHash());
}

} // namespace StorageUtilities
} // namespace WebCore
