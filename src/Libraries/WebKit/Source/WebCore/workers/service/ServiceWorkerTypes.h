/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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

#include "ProcessIdentifier.h"
#include "ProcessQualified.h"
#include "ScriptBuffer.h"
#include "ScriptExecutionContextIdentifier.h"
#include "ServiceWorkerIdentifier.h"
#include <variant>
#include <wtf/ObjectIdentifier.h>
#include <wtf/RobinHoodHashMap.h>
#include <wtf/URLHash.h>

namespace WebCore {

struct ServiceWorkerData;
struct ServiceWorkerClientData;

enum class ServiceWorkerRegistrationState : uint8_t {
    Installing = 0,
    Waiting = 1,
    Active = 2,
};

enum class ServiceWorkerState : uint8_t {
    Parsed,
    Installing,
    Installed,
    Activating,
    Activated,
    Redundant,
};

enum class ServiceWorkerClientFrameType : uint8_t {
    Auxiliary,
    TopLevel,
    Nested,
    None
};

enum class ServiceWorkerIsInspectable : bool { No, Yes };
enum class ShouldNotifyWhenResolved : bool { No, Yes };

enum class ServiceWorkerRegistrationIdentifierType { };
using ServiceWorkerRegistrationIdentifier = AtomicObjectIdentifier<ServiceWorkerRegistrationIdentifierType>;

enum class ServiceWorkerJobIdentifierType { };
using ServiceWorkerJobIdentifier = AtomicObjectIdentifier<ServiceWorkerJobIdentifierType>;

enum class SWServerToContextConnectionIdentifierType { };
using SWServerToContextConnectionIdentifier = ObjectIdentifier<SWServerToContextConnectionIdentifierType>;

using SWServerConnectionIdentifierType = ProcessIdentifierType;
using SWServerConnectionIdentifier = ObjectIdentifier<SWServerConnectionIdentifierType>;

using ServiceWorkerOrClientData = std::variant<ServiceWorkerData, ServiceWorkerClientData>;

// FIXME: It should be possible to replace ServiceWorkerOrClientIdentifier with ScriptExecutionContextIdentifier entirely.
using ServiceWorkerOrClientIdentifier = std::variant<ScriptExecutionContextIdentifier, ServiceWorkerIdentifier>;

struct ServiceWorkerScripts {
    ServiceWorkerScripts isolatedCopy() const
    {
        MemoryCompactRobinHoodHashMap<WTF::URL, ScriptBuffer> isolatedImportedScripts;
        for (auto& [url, script] : importedScripts)
            isolatedImportedScripts.add(url.isolatedCopy(), script.isolatedCopy());
        return { identifier, mainScript.isolatedCopy(), WTFMove(isolatedImportedScripts) };
    }

    ServiceWorkerIdentifier identifier;
    ScriptBuffer mainScript;
    MemoryCompactRobinHoodHashMap<WTF::URL, ScriptBuffer> importedScripts;
};

} // namespace WebCore
