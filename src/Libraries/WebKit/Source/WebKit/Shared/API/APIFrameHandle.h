/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 20, 2024.
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
#include <WebCore/FrameIdentifier.h>
#include <wtf/Ref.h>

namespace API {

class FrameHandle final : public ObjectImpl<Object::Type::FrameHandle> {
public:
    static Ref<FrameHandle> create(std::optional<WebCore::FrameIdentifier> frameID)
    {
        return adoptRef(*new FrameHandle(frameID, false));
    }
    static Ref<FrameHandle> createAutoconverting(WebCore::FrameIdentifier frameID)
    {
        return adoptRef(*new FrameHandle(frameID, true));
    }
    static Ref<FrameHandle> create(std::optional<WebCore::FrameIdentifier> frameID, bool autoconverting)
    {
        return adoptRef(*new FrameHandle(frameID, autoconverting));
    }

    explicit FrameHandle(std::optional<WebCore::FrameIdentifier> frameID, bool isAutoconverting)
        : m_frameID(frameID)
        , m_isAutoconverting(isAutoconverting)
    {
    }

    Markable<WebCore::FrameIdentifier> frameID() const { return m_frameID; }
    bool isAutoconverting() const { return m_isAutoconverting; }

private:
    const Markable<WebCore::FrameIdentifier> m_frameID;
    const bool m_isAutoconverting;
};

} // namespace API

SPECIALIZE_TYPE_TRAITS_API_OBJECT(FrameHandle);
