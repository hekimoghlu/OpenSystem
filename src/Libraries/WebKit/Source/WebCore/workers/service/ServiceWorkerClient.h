/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 14, 2024.
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

#include "ContextDestructionObserver.h"
#include "ExceptionOr.h"
#include "ScriptExecutionContextIdentifier.h"
#include "ServiceWorkerClientData.h"
#include <JavaScriptCore/Strong.h>
#include <wtf/RefCounted.h>

namespace JSC {
class JSGlobalObject;
class JSValue;
}

namespace WebCore {

class ServiceWorkerGlobalScope;

struct StructuredSerializeOptions;

class ServiceWorkerClient : public RefCounted<ServiceWorkerClient>, public ContextDestructionObserver {
public:
    using Identifier = ScriptExecutionContextIdentifier;

    using Type = ServiceWorkerClientType;
    using FrameType = ServiceWorkerClientFrameType;

    static Ref<ServiceWorkerClient> create(ServiceWorkerGlobalScope&, ServiceWorkerClientData&&);

    ~ServiceWorkerClient();

    const URL& url() const;
    FrameType frameType() const;
    Type type() const;
    String id() const;

    Identifier identifier() const { return m_data.identifier; }

    ExceptionOr<void> postMessage(JSC::JSGlobalObject&, JSC::JSValue message, StructuredSerializeOptions&&);

    const ServiceWorkerClientData& data() const { return m_data; }

protected:
    ServiceWorkerClient(ServiceWorkerGlobalScope&, ServiceWorkerClientData&&);

private:
    ServiceWorkerClientData m_data;
};

} // namespace WebCore
