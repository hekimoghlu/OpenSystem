/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 16, 2023.
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

#if ENABLE(WEB_RTC)

#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>
#include <wtf/text/WTFString.h>

namespace WebCore {

class RTCSessionDescriptionDescriptor;

class RTCSessionDescriptionRequest : public RefCounted<RTCSessionDescriptionRequest> {
public:
    class ExtraData : public RefCounted<ExtraData> {
    public:
        virtual ~ExtraData() = default;
    };

    virtual ~RTCSessionDescriptionRequest() = default;

    virtual void requestSucceeded(RTCSessionDescriptionDescriptor&) = 0;
    virtual void requestFailed(const String& error) = 0;

    ExtraData* extraData() const { return m_extraData.get(); }
    void setExtraData(RefPtr<ExtraData>&& extraData) { m_extraData = extraData; }

protected:
    RTCSessionDescriptionRequest() = default;

private:
    RefPtr<ExtraData> m_extraData;
};

} // namespace WebCore

#endif // ENABLE(WEB_RTC)
