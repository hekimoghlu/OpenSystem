/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, June 21, 2023.
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

#include "CreateHTMLCallback.h"
#include "CreateScriptCallback.h"
#include "CreateScriptURLCallback.h"
#include "ExceptionOr.h"
#include "ScriptWrappable.h"
#include "TrustedTypePolicyOptions.h"
#include <JavaScriptCore/Strong.h>
#include <wtf/Ref.h>
#include <wtf/RefCounted.h>

namespace WebCore {

class TrustedHTML;
class TrustedScript;
class TrustedScriptURL;
enum class TrustedType : int8_t;
struct TrustedTypePolicyOptions;
class WebCoreOpaqueRoot;

enum class IfMissing : bool { Throw, ReturnNull };

class TrustedTypePolicy : public ScriptWrappable, public RefCounted<TrustedTypePolicy> {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TrustedTypePolicy);
public:
    static Ref<TrustedTypePolicy> create(const String&, const TrustedTypePolicyOptions&);
    ~TrustedTypePolicy() = default;
    ExceptionOr<Ref<TrustedHTML>> createHTML(const String& input, FixedVector<JSC::Strong<JSC::Unknown>>&&);
    ExceptionOr<Ref<TrustedScript>> createScript(const String& input, FixedVector<JSC::Strong<JSC::Unknown>>&&);
    ExceptionOr<Ref<TrustedScriptURL>> createScriptURL(const String& input, FixedVector<JSC::Strong<JSC::Unknown>>&&);
    ExceptionOr<String> getPolicyValue(TrustedType trustedTypeName, const String& input, FixedVector<JSC::Strong<JSC::Unknown>>&&, IfMissing = IfMissing::Throw);
    const String name() const { return m_name; }

    const TrustedTypePolicyOptions& options() const
    {
        IGNORE_CLANG_WARNINGS_BEGIN("thread-safety-reference-return")
        return m_options;
        IGNORE_CLANG_WARNINGS_END
    }
    Lock& lock() WTF_RETURNS_LOCK(m_lock) { return m_lock; }

private:
    TrustedTypePolicy(const String&, const TrustedTypePolicyOptions&);

    String m_name;
    TrustedTypePolicyOptions m_options WTF_GUARDED_BY_LOCK(m_lock);
    mutable Lock m_lock;
};

WebCoreOpaqueRoot root(TrustedTypePolicy*);

} // namespace WebCore
