/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 20, 2023.
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
#if JSC_OBJC_API_ENABLED

#import "SourceProvider.h"

@class JSScript;

class JSScriptSourceProvider final : public JSC::SourceProvider {
public:
    template<typename... Args>
    static Ref<JSScriptSourceProvider> create(JSScript *script, Args&&... args)
    {
        return adoptRef(*new JSScriptSourceProvider(script, std::forward<Args>(args)...));
    }

    unsigned hash() const final;
    StringView source() const final;
    RefPtr<JSC::CachedBytecode> cachedBytecode() const final;

private:
    template<typename... Args>
    JSScriptSourceProvider(JSScript *script, Args&&... args)
        : SourceProvider(std::forward<Args>(args)...)
        , m_script(script)
    { }

    RetainPtr<JSScript> m_script;
};

#endif // JSC_OBJC_API_ENABLED
