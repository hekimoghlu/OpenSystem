/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, April 9, 2025.
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

#include <JavaScriptCore/Strong.h>
#include <wtf/HashMap.h>
#include <wtf/TZoneMalloc.h>

namespace WebCore {

class DOMWrapperWorld;
class JSDOMWindow;
class LocalFrame;

class ScriptCachedFrameData {
    WTF_MAKE_TZONE_ALLOCATED(ScriptCachedFrameData);
    WTF_MAKE_NONCOPYABLE(ScriptCachedFrameData);
public:
    explicit ScriptCachedFrameData(LocalFrame&);
    ~ScriptCachedFrameData();

    void restore(LocalFrame&);
    void clear();

private:
    typedef UncheckedKeyHashMap<RefPtr<DOMWrapperWorld>, JSC::Strong<JSDOMWindow>> JSDOMWindowSet;
    JSDOMWindowSet m_windows;
};

} // namespace WebCore
