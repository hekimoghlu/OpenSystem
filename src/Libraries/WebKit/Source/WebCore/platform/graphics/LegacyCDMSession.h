/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, June 10, 2025.
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

#if ENABLE(LEGACY_ENCRYPTED_MEDIA)

#include <JavaScriptCore/Forward.h>
#include <wtf/AbstractRefCounted.h>
#include <wtf/Forward.h>
#include <wtf/TZoneMallocInlines.h>
#include <wtf/WeakPtr.h>

namespace WebCore {
class LegacyCDMSessionClient;
}

namespace WTF {
template<typename T> struct IsDeprecatedWeakRefSmartPointerException;
template<> struct IsDeprecatedWeakRefSmartPointerException<WebCore::LegacyCDMSessionClient> : std::true_type { };
}

namespace WebCore {

class LegacyCDMSessionClient : public CanMakeWeakPtr<LegacyCDMSessionClient> {
    WTF_MAKE_TZONE_ALLOCATED_INLINE(LegacyCDMSessionClient);
public:
    virtual ~LegacyCDMSessionClient() = default;
    virtual void sendMessage(Uint8Array*, String destinationURL) = 0;

    enum : uint8_t {
        MediaKeyErrorUnknown = 1,
        MediaKeyErrorClient,
        MediaKeyErrorService,
        MediaKeyErrorOutput,
        MediaKeyErrorHardwareChange,
        MediaKeyErrorDomain,
    };
    typedef unsigned short MediaKeyErrorCode;
    virtual void sendError(MediaKeyErrorCode, uint32_t systemCode) = 0;

    virtual String mediaKeysStorageDirectory() const = 0;

#if !RELEASE_LOG_DISABLED
    virtual const Logger& logger() const = 0;
    virtual uint64_t logIdentifier() const = 0;
#endif
};

enum LegacyCDMSessionType {
    CDMSessionTypeUnknown,
    CDMSessionTypeClearKey,
    CDMSessionTypeAVFoundationObjC,
    CDMSessionTypeAVContentKeySession,
    CDMSessionTypeRemote,
};

class WEBCORE_EXPORT LegacyCDMSession : public AbstractRefCounted {
public:
    virtual ~LegacyCDMSession() = default;
    virtual void invalidate() { }

    virtual LegacyCDMSessionType type() { return CDMSessionTypeUnknown; }
    virtual const String& sessionId() const = 0;
    virtual RefPtr<Uint8Array> generateKeyRequest(const String& mimeType, Uint8Array* initData, String& destinationURL, unsigned short& errorCode, uint32_t& systemCode) = 0;
    virtual void releaseKeys() = 0;
    virtual bool update(Uint8Array*, RefPtr<Uint8Array>& nextMessage, unsigned short& errorCode, uint32_t& systemCode) = 0;
    virtual RefPtr<ArrayBuffer> cachedKeyForKeyID(const String&) const = 0;
};

}

#endif // ENABLE(LEGACY_ENCRYPTED_MEDIA)
