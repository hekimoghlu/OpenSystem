/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 20, 2022.
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

#include "XMLHttpRequest.h"
#include <wtf/Forward.h>

namespace WebCore {

class WebCoreOpaqueRoot;

class XMLHttpRequestUpload final : public XMLHttpRequestEventTarget {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(XMLHttpRequestUpload);
public:
    explicit XMLHttpRequestUpload(XMLHttpRequest&);

    void ref() { m_request.ref(); }
    void deref() { m_request.deref(); }

    void dispatchProgressEvent(const AtomString& type, unsigned long long loaded, unsigned long long total);

    bool hasRelevantEventListener() const;

private:
    // EventTarget.
    void eventListenersDidChange() final;
    void refEventTarget() final { ref(); }
    void derefEventTarget() final { deref(); }

    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::XMLHttpRequestUpload; }
    ScriptExecutionContext* scriptExecutionContext() const final { return m_request.scriptExecutionContext(); }

    XMLHttpRequest& m_request;
};

WebCoreOpaqueRoot root(XMLHttpRequestUpload*);
    
} // namespace WebCore
