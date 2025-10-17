/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, October 14, 2022.
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

#include <wtf/AbstractRefCounted.h>
#include <wtf/Ref.h>

namespace WebCore {
    
enum class AudioHardwareActivityType {
    Unknown,
    IsActive,
    IsInactive
};

class AudioHardwareListener : public AbstractRefCounted {
public:
    class Client {
    public:
        virtual ~Client() = default;
        virtual void audioHardwareDidBecomeActive() = 0;
        virtual void audioHardwareDidBecomeInactive() = 0;
        virtual void audioOutputDeviceChanged() = 0;
    };

    using CreationFunction = Function<Ref<AudioHardwareListener>(AudioHardwareListener::Client&)>;
    WEBCORE_EXPORT static void setCreationFunction(CreationFunction&&);
    WEBCORE_EXPORT static void resetCreationFunction();

    WEBCORE_EXPORT static Ref<AudioHardwareListener> create(Client&);
    virtual ~AudioHardwareListener() = default;
    
    AudioHardwareActivityType hardwareActivity() const { return m_activity; }

    struct BufferSizeRange {
        size_t minimum { 0 };
        size_t maximum { 0 };
        operator bool() const { return minimum && maximum; }
        size_t nearest(size_t value) const { return std::min(std::max(value, minimum), maximum); }
    };
    BufferSizeRange supportedBufferSizes() const { return m_supportedBufferSizes; }

protected:
    WEBCORE_EXPORT AudioHardwareListener(Client&);

    void setHardwareActivity(AudioHardwareActivityType activity) { m_activity = activity; }
    void setSupportedBufferSizes(BufferSizeRange sizes) { m_supportedBufferSizes = sizes; }

    Client& m_client;
    AudioHardwareActivityType m_activity { AudioHardwareActivityType::Unknown };
    BufferSizeRange m_supportedBufferSizes;
};

}
