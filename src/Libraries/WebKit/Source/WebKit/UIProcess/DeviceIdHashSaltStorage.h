/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 29, 2023.
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

#include "UserMediaPermissionCheckProxy.h"
#include <WebCore/KeyedCoding.h>
#include <WebCore/SecurityOrigin.h>
#include <wtf/CompletionHandler.h>
#include <wtf/HashMap.h>
#include <wtf/HashSet.h>
#include <wtf/Ref.h>
#include <wtf/WorkQueue.h>

namespace WebKit {

class DeviceIdHashSaltStorage : public ThreadSafeRefCounted<DeviceIdHashSaltStorage, WTF::DestructionThread::MainRunLoop> {
public:
    static Ref<DeviceIdHashSaltStorage> create(const String& deviceIdHashSaltStorageDirectory);
    ~DeviceIdHashSaltStorage();

    void deviceIdHashSaltForOrigin(const WebCore::SecurityOrigin& documentOrigin, const WebCore::SecurityOrigin& parentOrigin, CompletionHandler<void(String&&)>&&);

    void getDeviceIdHashSaltOrigins(CompletionHandler<void(HashSet<WebCore::SecurityOriginData>&&)>&&);
    void deleteDeviceIdHashSaltForOrigins(const Vector<WebCore::SecurityOriginData>&, CompletionHandler<void()>&&);
    void deleteDeviceIdHashSaltOriginsModifiedSince(WallTime, CompletionHandler<void()>&&);

private:
    struct HashSaltForOrigin {
        WTF_MAKE_STRUCT_FAST_ALLOCATED;

        HashSaltForOrigin(WebCore::SecurityOriginData&& documentOrigin, WebCore::SecurityOriginData&& parentOrigin, String&& deviceIdHashSalt, WallTime lastTimeUsed = WallTime::now())
            : documentOrigin(WTFMove(documentOrigin))
            , parentOrigin(WTFMove(parentOrigin))
            , deviceIdHashSalt(WTFMove(deviceIdHashSalt))
            , lastTimeUsed(lastTimeUsed)
        { }

        HashSaltForOrigin isolatedCopy() const & { return { documentOrigin.isolatedCopy(), parentOrigin.isolatedCopy(), deviceIdHashSalt.isolatedCopy(), lastTimeUsed }; }
        HashSaltForOrigin isolatedCopy() && { return { WTFMove(documentOrigin).isolatedCopy(), WTFMove(parentOrigin).isolatedCopy(), WTFMove(deviceIdHashSalt).isolatedCopy(), lastTimeUsed }; }

        WebCore::SecurityOriginData documentOrigin;
        WebCore::SecurityOriginData parentOrigin;
        String deviceIdHashSalt;
        WallTime lastTimeUsed;
    };

    DeviceIdHashSaltStorage(const String& deviceIdHashSaltStorageDirectory);
    void loadStorageFromDisk(CompletionHandler<void(HashMap<String, std::unique_ptr<HashSaltForOrigin>>&&)>&&);
    void storeHashSaltToDisk(const HashSaltForOrigin&);
    void deleteHashSaltFromDisk(const HashSaltForOrigin&);
    std::unique_ptr<WebCore::KeyedEncoder> createEncoderFromData(const HashSaltForOrigin&) const;
    std::unique_ptr<HashSaltForOrigin> getDataFromDecoder(WebCore::KeyedDecoder*, String&& deviceIdHashSalt) const;
    void completePendingHandler(CompletionHandler<void(HashSet<WebCore::SecurityOriginData>&&)>&&);
    void completeDeviceIdHashSaltForOriginCall(WebCore::SecurityOriginData&& documentOrigin, WebCore::SecurityOriginData&& parentOrigin, CompletionHandler<void(String&&)>&&);

    Ref<WorkQueue> m_queue;
    HashMap<String, std::unique_ptr<HashSaltForOrigin>> m_deviceIdHashSaltForOrigins;
    bool m_isLoaded { false };
    bool m_isClosed { false };
    Vector<CompletionHandler<void()>> m_pendingCompletionHandlers;
    const String m_deviceIdHashSaltStorageDirectory;
};

} // namespace WebKit
