/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, September 29, 2022.
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

#if ENABLE(CONTENT_EXTENSIONS)

#include "DFABytecodeInterpreter.h"
#include "StyleSheetContents.h"
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {

namespace ContentExtensions {

class CompiledContentExtension;

class ContentExtension : public RefCounted<ContentExtension> {
public:
    enum class ShouldCompileCSS : bool { No, Yes };
    static Ref<ContentExtension> create(const String& identifier, Ref<CompiledContentExtension>&&, URL&&, ShouldCompileCSS);

    const String& identifier() const { return m_identifier; }
    const URL& extensionBaseURL() const { return m_extensionBaseURL; }
    const CompiledContentExtension& compiledExtension() const { return m_compiledExtension.get(); }
    StyleSheetContents* globalDisplayNoneStyleSheet();
    const DFABytecodeInterpreter::Actions& topURLActions(const URL& topURL) const;
    const DFABytecodeInterpreter::Actions& frameURLActions(const URL& frameURL) const;
    const Vector<uint64_t>& universalActions() const { return m_universalActions; }

private:
    ContentExtension(const String& identifier, Ref<CompiledContentExtension>&&, URL&&, ShouldCompileCSS);
    uint32_t findFirstIgnorePreviousRules() const;
    
    String m_identifier;
    Ref<CompiledContentExtension> m_compiledExtension;
    URL m_extensionBaseURL;

    RefPtr<StyleSheetContents> m_globalDisplayNoneStyleSheet;
    void compileGlobalDisplayNoneStyleSheet();

    void populateTopURLActionCacheIfNeeded(const URL& topURL) const;
    mutable URL m_cachedTopURL;
    mutable DFABytecodeInterpreter::Actions m_cachedTopURLActions;

    void populateFrameURLActionCacheIfNeeded(const URL& frameURL) const;
    mutable URL m_cachedFrameURL;
    mutable DFABytecodeInterpreter::Actions m_cachedFrameURLActions;

    Vector<uint64_t> m_universalActions;
};

} // namespace ContentExtensions
} // namespace WebCore

#endif // ENABLE(CONTENT_EXTENSIONS)
