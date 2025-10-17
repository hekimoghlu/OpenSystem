/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 3, 2023.
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

#include <wtf/Noncopyable.h>
#include <wtf/TZoneMalloc.h>
#include <wtf/text/WTFString.h>

#if USE(CF)
#include <wtf/RetainPtr.h>
#endif

#if USE(GLIB)
typedef struct _GModule GModule;
#endif

#if OS(WINDOWS)
#include <windows.h>
#endif

namespace WebKit {

class Module {
    WTF_MAKE_TZONE_ALLOCATED(Module);
    WTF_MAKE_NONCOPYABLE(Module);
public:
    explicit Module(const String& path);
    ~Module();

    bool load();
    // Note: On Mac this leaks the CFBundle to avoid crashes when a bundle is unloaded and there are
    // live Objective-C objects whose methods come from that bundle.
    void unload();

#if USE(CF)
    String bundleIdentifier() const;
#endif

    template<typename FunctionType> FunctionType functionPointer(const char* functionName) const;

private:
    void* platformFunctionPointer(const char* functionName) const;

    String m_path;
#if OS(WINDOWS)
    HMODULE m_module;
#endif
#if USE(CF)
    RetainPtr<CFBundleRef> m_bundle;
#elif USE(GLIB)
    GModule* m_handle;
#endif
};

template<typename FunctionType> FunctionType Module::functionPointer(const char* functionName) const
{
    return reinterpret_cast<FunctionType>(platformFunctionPointer(functionName));
}

}
