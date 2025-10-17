/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 28, 2023.
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

#if ENABLE(MEDIA_SOURCE)

#include "ActiveDOMObject.h"
#include "EventTarget.h"
#include <wtf/RefCounted.h>
#include <wtf/Vector.h>

namespace WebCore {

class SourceBuffer;
class WebCoreOpaqueRoot;

class SourceBufferList final : public RefCounted<SourceBufferList>, public EventTarget, public ActiveDOMObject {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(SourceBufferList);
public:
    void ref() const final { RefCounted::ref(); }
    void deref() const final { RefCounted::deref(); }

    static Ref<SourceBufferList> create(ScriptExecutionContext*);
    virtual ~SourceBufferList();

    bool isSupportedPropertyIndex(unsigned index) const { return index < length(); }
    unsigned length() const { return m_list.size(); }

    RefPtr<SourceBuffer> item(unsigned index) const;

    void add(Ref<SourceBuffer>&&);
    void remove(SourceBuffer&);
    bool contains(SourceBuffer&) const;
    void clear();
    void replaceWith(Vector<Ref<SourceBuffer>>&&);

    auto begin() { return m_list.begin(); }
    auto end() { return m_list.end(); }
    auto begin() const { return m_list.begin(); }
    auto end() const { return m_list.end(); }
    size_t size() const { return m_list.size(); }

    // EventTarget interface
    enum EventTargetInterfaceType eventTargetInterface() const final { return EventTargetInterfaceType::SourceBufferList; }
    ScriptExecutionContext* scriptExecutionContext() const final { return ContextDestructionObserver::scriptExecutionContext(); }

private:
    explicit SourceBufferList(ScriptExecutionContext*);

    void scheduleEvent(const AtomString&);

    void refEventTarget() override { ref(); }
    void derefEventTarget() override { deref(); }

    Vector<Ref<SourceBuffer>> m_list;
};

WebCoreOpaqueRoot root(SourceBufferList*);

} // namespace WebCore

#endif // ENABLE(MEDIA_SOURCE)
