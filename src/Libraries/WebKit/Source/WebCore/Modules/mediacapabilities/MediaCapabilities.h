/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 27, 2022.
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

#include "MediaEngineConfigurationFactory.h"
#include <wtf/HashMap.h>
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class DeferredPromise;
class ScriptExecutionContext;

class MediaCapabilities : public RefCountedAndCanMakeWeakPtr<MediaCapabilities> {
public:
    static Ref<MediaCapabilities> create() { return adoptRef(*new MediaCapabilities); }

    void decodingInfo(ScriptExecutionContext&, MediaDecodingConfiguration&&, Ref<DeferredPromise>&&);
    void encodingInfo(ScriptExecutionContext&, MediaEncodingConfiguration&&, Ref<DeferredPromise>&&);

private:
    MediaCapabilities() = default;

    uint64_t m_nextTaskIdentifier { 0 };
    HashMap<uint64_t, MediaEngineConfigurationFactory::DecodingConfigurationCallback> m_decodingTasks;
    HashMap<uint64_t, MediaEngineConfigurationFactory::EncodingConfigurationCallback> m_encodingTasks;
};

} // namespace WebCore
