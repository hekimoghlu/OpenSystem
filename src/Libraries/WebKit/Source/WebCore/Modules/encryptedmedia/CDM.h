/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 17, 2024.
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

#if ENABLE(ENCRYPTED_MEDIA)

#include "CDMPrivate.h"
#include "ContextDestructionObserver.h"
#include "MediaKeySessionType.h"
#include "MediaKeySystemConfiguration.h"
#include "MediaKeySystemMediaCapability.h"
#include "MediaKeysRestrictions.h"
#include <wtf/Function.h>
#include <wtf/Ref.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>
#include <wtf/text/WTFString.h>

#if !RELEASE_LOG_DISABLED
namespace WTF {
class Logger;
}
#endif

namespace WebCore {

class CDMFactory;
class CDMInstance;
class CDMPrivate;
class Document;
class ScriptExecutionContext;
class SharedBuffer;

class CDM : public RefCountedAndCanMakeWeakPtr<CDM>, public CDMPrivateClient, private ContextDestructionObserver {
public:
    static bool supportsKeySystem(const String&);
    static bool isPersistentType(MediaKeySessionType);

    static Ref<CDM> create(Document&, const String& keySystem);
    ~CDM();

    using SupportedConfigurationCallback = Function<void(std::optional<MediaKeySystemConfiguration>)>;
    void getSupportedConfiguration(MediaKeySystemConfiguration&& candidateConfiguration, SupportedConfigurationCallback&&);

    const String& keySystem() const { return m_keySystem; }

    void loadAndInitialize();
    RefPtr<CDMInstance> createInstance();
    bool supportsServerCertificates() const;
    bool supportsSessions() const;
    bool supportsInitDataType(const AtomString&) const;

    RefPtr<SharedBuffer> sanitizeInitData(const AtomString& initDataType, const SharedBuffer&);
    bool supportsInitData(const AtomString& initDataType, const SharedBuffer&);

    RefPtr<SharedBuffer> sanitizeResponse(const SharedBuffer&);

    std::optional<String> sanitizeSessionId(const String& sessionId);

    String storageDirectory() const;

#if !RELEASE_LOG_DISABLED
    const Logger& logger() const final { return m_logger; }
    uint64_t logIdentifier() const { return m_logIdentifier; }
#endif

private:
    CDM(Document&, const String& keySystem);

#if !RELEASE_LOG_DISABLED
    Ref<const Logger> m_logger;
    const uint64_t m_logIdentifier;
#endif
    String m_keySystem;
    std::unique_ptr<CDMPrivate> m_private;
};

}

#endif
