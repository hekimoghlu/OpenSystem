/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 6, 2022.
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

#if ENABLE(WEBGL)

#include "ContextDestructionObserver.h"
#include "WebGLObject.h"
#include <wtf/HashMap.h>
#include <wtf/HashFunctions.h>
#include <wtf/Lock.h>
#include <wtf/Vector.h>

namespace JSC {
class AbstractSlotVisitor;
}

namespace WTF {
class AbstractLocker;
};

namespace WebCore {

class ScriptExecutionContext;
class WebGLRenderingContextBase;
class WebGLShader;

class WebGLProgram final : public WebGLObject, public ContextDestructionObserver {
public:
    static RefPtr<WebGLProgram> create(WebGLRenderingContextBase&);
    virtual ~WebGLProgram();

    static UncheckedKeyHashMap<WebGLProgram*, WebGLRenderingContextBase*>& instances() WTF_REQUIRES_LOCK(instancesLock());
    static Lock& instancesLock() WTF_RETURNS_LOCK(s_instancesLock);

    void contextDestroyed() final;

    bool getLinkStatus();

    unsigned getLinkCount() const { return m_linkCount; }

    // This is to be called everytime after the program is successfully linked.
    // We don't deal with integer overflow here, assuming in reality a program
    // will never be linked so many times.
    // Also, we invalidate the cached program info.
    void increaseLinkCount();

    WebGLShader* getAttachedShader(GCGLenum);
    bool attachShader(const AbstractLocker&, WebGLShader*);
    bool detachShader(const AbstractLocker&, WebGLShader*);
    
    void setRequiredTransformFeedbackBufferCount(int count)
    {
        m_requiredTransformFeedbackBufferCountAfterNextLink = count;
    }
    int requiredTransformFeedbackBufferCount()
    {
        cacheInfoIfNeeded();
        return m_requiredTransformFeedbackBufferCount;
    }

    void addMembersToOpaqueRoots(const AbstractLocker&, JSC::AbstractSlotVisitor&);

    bool isUsable() const { return object(); }
    bool isInitialized() const { return true; }

private:
    WebGLProgram(WebGLRenderingContextBase&, PlatformGLObject);

    void deleteObjectImpl(const AbstractLocker&, GraphicsContextGL*, PlatformGLObject) override;

    void cacheInfoIfNeeded();

    static Lock s_instancesLock;

    GCGLint m_linkStatus { 0 };

    // This is used to track whether a WebGLUniformLocation belongs to this program or not.
    unsigned m_linkCount { 0 };

    RefPtr<WebGLShader> m_vertexShader;
    RefPtr<WebGLShader> m_fragmentShader;

    bool m_infoValid { true };
    int m_requiredTransformFeedbackBufferCountAfterNextLink { 0 };
    int m_requiredTransformFeedbackBufferCount { 0 };
};

} // namespace WebCore

#endif
