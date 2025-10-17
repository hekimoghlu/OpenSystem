/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, February 28, 2025.
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

#if ENABLE(VIDEO)

#include "SerializedPlatformDataCueValue.h"
#include <JavaScriptCore/ArrayBuffer.h>
#include <JavaScriptCore/JSCJSValue.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class SerializedPlatformDataCue : public RefCounted<SerializedPlatformDataCue> {
public:
    WEBCORE_EXPORT static Ref<SerializedPlatformDataCue> create(SerializedPlatformDataCueValue&&);

    virtual ~SerializedPlatformDataCue() = default;

    virtual JSC::JSValue deserialize(JSC::JSGlobalObject*) const { return JSC::jsNull(); }
    virtual RefPtr<JSC::ArrayBuffer> data() const { return { }; }
    virtual bool isEqual(const SerializedPlatformDataCue&) const { return false; }

    enum class PlatformType : bool { None, ObjC };
    virtual PlatformType platformType() const { return PlatformType::None; }

    virtual bool encodingRequiresPlatformData() const { return false; }

    virtual SerializedPlatformDataCueValue encodableValue() const { return { }; }

protected:
    SerializedPlatformDataCue() = default;
};

} // namespace WebCore

#endif
