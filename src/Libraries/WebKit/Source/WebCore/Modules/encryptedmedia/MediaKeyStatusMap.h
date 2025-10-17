/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 10, 2025.
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

#if ENABLE(ENCRYPTED_MEDIA)

#include "BufferSource.h"
#include "MediaKeyStatus.h"
#include <JavaScriptCore/JSCJSValueInlines.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class MediaKeySession;
class ScriptExecutionContext;
class SharedBuffer;

class MediaKeyStatusMap : public RefCounted<MediaKeyStatusMap> {
public:
    using Status = MediaKeyStatus;

    static Ref<MediaKeyStatusMap> create(const MediaKeySession& session)
    {
        return adoptRef(*new MediaKeyStatusMap(session));
    }

    virtual ~MediaKeyStatusMap();

    void detachSession();

    unsigned long size();
    bool has(const BufferSource&);
    JSC::JSValue get(JSC::JSGlobalObject&, const BufferSource&);

    class Iterator {
    public:
        explicit Iterator(MediaKeyStatusMap&);
        std::optional<KeyValuePair<BufferSource::VariantType, MediaKeyStatus>> next();

    private:
        Ref<MediaKeyStatusMap> m_map;
        size_t m_index { 0 };
    };
    Iterator createIterator(ScriptExecutionContext*) { return Iterator(*this); }

private:
    MediaKeyStatusMap(const MediaKeySession&);

    const MediaKeySession* m_session;
};

} // namespace WebCore

#endif // ENABLE(ENCRYPTED_MEDIA)
