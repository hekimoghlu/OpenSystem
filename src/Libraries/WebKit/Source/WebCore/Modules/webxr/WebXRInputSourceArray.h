/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, December 11, 2024.
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

#if ENABLE(WEBXR)

#include "PlatformXR.h"
#include "ScriptWrappable.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/UniqueRef.h>

namespace WebCore {

class Document;
class WebXRInputSource;
class XRInputSourceEvent;
class WebXRSession;

class WebXRInputSourceArray final : public ScriptWrappable {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(WebXRInputSourceArray);
public:
    explicit WebXRInputSourceArray(WebXRSession&);

    using InputSourceList = Vector<PlatformXR::FrameData::InputSource>;
    static UniqueRef<WebXRInputSourceArray> create(WebXRSession&);
    ~WebXRInputSourceArray();

    void ref();
    void deref();

    unsigned length() const;
    bool isSupportedPropertyIndex(unsigned index) const { return index < length(); }
    WebXRInputSource* item(unsigned) const;

    void clear();
    void update(double timestamp, const InputSourceList&);

    // For GC reachablitiy.
    WebXRSession* session() const { return &m_session; }

private:
    void handleRemovedInputSources(const InputSourceList&, Vector<Ref<WebXRInputSource>>&, Vector<Ref<WebXRInputSource>>&, Vector<Ref<XRInputSourceEvent>>&);
    void handleAddedOrUpdatedInputSources(double timestamp, const InputSourceList&, Vector<Ref<WebXRInputSource>>&, Vector<Ref<WebXRInputSource>>&, Vector<Ref<WebXRInputSource>>&, Vector<Ref<XRInputSourceEvent>>&);

    WebXRSession& m_session;
    Vector<Ref<WebXRInputSource>> m_inputSources;
};

} // namespace WebCore

#endif // ENABLE(WEBXR)
