/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 28, 2025.
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
#include <WebCore/ClientOrigin.h>
#include <wtf/CompletionHandler.h>

namespace WebKit {

class MediaKeySystemPermissionRequest : public RefCounted<MediaKeySystemPermissionRequest> {
public:
    static Ref<MediaKeySystemPermissionRequest> create(const String& keySystem, CompletionHandler<void(bool)>&& completionHandler)
    {
        return adoptRef(*new MediaKeySystemPermissionRequest(keySystem, WTFMove(completionHandler)));
    }

    void complete(bool success)
    {
        auto completionHandler = std::exchange(m_completionHandler, { });
        completionHandler(success);
    }

    const String& keySystem() const { return m_keySystem; }

private:
    MediaKeySystemPermissionRequest(const String& keySystem, CompletionHandler<void(bool)>&& completionHandler)
        : m_keySystem(keySystem)
        , m_completionHandler(WTFMove(completionHandler))
    { }

    String m_keySystem;
    CompletionHandler<void(bool)> m_completionHandler;
};

class MediaKeySystemPermissionCallback : public API::ObjectImpl<API::Object::Type::MediaKeySystemPermissionCallback> {
public:
    static Ref<MediaKeySystemPermissionCallback> create(CompletionHandler<void(bool)>&& completionHandler)
    {
        return adoptRef(*new MediaKeySystemPermissionCallback(WTFMove(completionHandler)));
    }

    void complete(bool granted) { m_completionHandler(granted); }

private:
    MediaKeySystemPermissionCallback(CompletionHandler<void(bool)>&& completionHandler)
        : m_completionHandler(WTFMove(completionHandler))
    { }

    CompletionHandler<void(bool)> m_completionHandler;
};

} // namespace WebKit
