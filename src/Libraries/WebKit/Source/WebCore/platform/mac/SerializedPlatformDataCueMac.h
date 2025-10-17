/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 17, 2023.
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

#if ENABLE(VIDEO) && ENABLE(DATACUE_VALUE)

#include "SerializedPlatformDataCue.h"
#include "SerializedPlatformDataCueValue.h"
#include <wtf/HashSet.h>

namespace WebCore {

class SerializedPlatformDataCueMac final : public SerializedPlatformDataCue {
public:
    SerializedPlatformDataCueMac(SerializedPlatformDataCueValue&&);
    virtual ~SerializedPlatformDataCueMac() = default;

    JSC::JSValue deserialize(JSC::JSGlobalObject*) const final;
    RefPtr<ArrayBuffer> data() const final;
    bool isEqual(const SerializedPlatformDataCue&) const final;
    PlatformType platformType() const final { return PlatformType::ObjC; }
    bool encodingRequiresPlatformData() const final { return true; }

    WEBCORE_EXPORT SerializedPlatformDataCueValue encodableValue() const final;

    WEBCORE_EXPORT static const UncheckedKeyHashSet<RetainPtr<Class>>& allowedClassesForNativeValues();

private:
    SerializedPlatformDataCueValue m_value;
};

SerializedPlatformDataCueMac* toSerializedPlatformDataCueMac(SerializedPlatformDataCue*);
const SerializedPlatformDataCueMac* toSerializedPlatformDataCueMac(const SerializedPlatformDataCue*);

} // namespace WebCore

#endif
