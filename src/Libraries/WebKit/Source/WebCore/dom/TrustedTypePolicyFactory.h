/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 21, 2021.
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

#include "ContextDestructionObserver.h"
#include "ScriptWrappable.h"
#include "TrustedTypePolicy.h"
#include <wtf/Forward.h>
#include <wtf/ListHashSet.h>
#include <wtf/RefCounted.h>
#include <wtf/RefPtr.h>

namespace WebCore {

class TrustedHTML;
class TrustedScript;
struct TrustedTypePolicyOptions;
class ScriptExecutionContext;

class TrustedTypePolicyFactory : public ScriptWrappable, public RefCounted<TrustedTypePolicyFactory>, public ContextDestructionObserver {
    WTF_MAKE_TZONE_OR_ISO_ALLOCATED(TrustedTypePolicyFactory);
public:
    static Ref<TrustedTypePolicyFactory> create(ScriptExecutionContext&);
    ~TrustedTypePolicyFactory() = default;

    ExceptionOr<Ref<TrustedTypePolicy>> createPolicy(ScriptExecutionContext&, const String& policyName, const TrustedTypePolicyOptions&);
    bool isHTML(JSC::JSValue) const;
    bool isScript(JSC::JSValue) const;
    bool isScriptURL(JSC::JSValue) const;
    Ref<TrustedHTML> emptyHTML() const;
    Ref<TrustedScript> emptyScript() const;

    String getAttributeType(const String& tagName, const String& attribute, const String& elementNamespace, const String& attributeNamespace) const;
    String getPropertyType(const String& tagName, const String& property, const String& elementNamespace) const;

    RefPtr<TrustedTypePolicy> defaultPolicy() const { return m_defaultPolicy; }
    TrustedTypePolicy* defaultPolicyConcurrently() const { return m_defaultPolicy.get(); }

private:
    TrustedTypePolicyFactory(ScriptExecutionContext&);

    RefPtr<TrustedTypePolicy> m_defaultPolicy;
    ListHashSet<String> m_createdPolicyNames;
};

} // namespace WebCore
