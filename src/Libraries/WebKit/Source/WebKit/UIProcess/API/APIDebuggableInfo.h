/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 31, 2024.
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

#include "APIObject.h"
#include "DebuggableInfoData.h"
#include <WebCore/InspectorDebuggableType.h>

namespace API {

class DebuggableInfo final : public ObjectImpl<Object::Type::DebuggableInfo> {
public:
    static Ref<DebuggableInfo> create(const WebKit::DebuggableInfoData&);
    DebuggableInfo() = default;
    virtual ~DebuggableInfo();

    Inspector::DebuggableType debuggableType() const { return m_data.debuggableType; }
    void setDebuggableType(Inspector::DebuggableType debuggableType) { m_data.debuggableType = debuggableType; }

    const WTF::String& targetPlatformName() const { return m_data.targetPlatformName; }
    void setTargetPlatformName(const WTF::String& targetPlatformName) { m_data.targetPlatformName = targetPlatformName; }

    const WTF::String& targetBuildVersion() const { return m_data.targetBuildVersion; }
    void setTargetBuildVersion(const WTF::String& targetBuildVersion) { m_data.targetBuildVersion = targetBuildVersion; }

    const WTF::String& targetProductVersion() const { return m_data.targetProductVersion; }
    void setTargetProductVersion(const WTF::String& targetProductVersion) { m_data.targetProductVersion = targetProductVersion; }

    bool targetIsSimulator() const { return m_data.targetIsSimulator; }
    void setTargetIsSimulator(bool targetIsSimulator) { m_data.targetIsSimulator = targetIsSimulator; }

    const WebKit::DebuggableInfoData& debuggableInfoData() const { return m_data; }

private:
    explicit DebuggableInfo(const WebKit::DebuggableInfoData&);

    WebKit::DebuggableInfoData m_data;
};

} // namespace API

SPECIALIZE_TYPE_TRAITS_API_OBJECT(DebuggableInfo);
