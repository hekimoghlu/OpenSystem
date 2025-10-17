/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, August 30, 2023.
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
#include "config.h"
#include "TrustedTypePolicyFactory.h"

#include "ContentSecurityPolicy.h"
#include "ContextDestructionObserver.h"
#include "HTMLNames.h"
#include "JSDOMConvertObject.h"
#include "JSTrustedHTML.h"
#include "JSTrustedScript.h"
#include "JSTrustedScriptURL.h"
#include "SVGNames.h"
#include "ScriptExecutionContext.h"
#include "TrustedType.h"
#include "TrustedTypePolicyOptions.h"
#include "XLinkNames.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(TrustedTypePolicyFactory);

Ref<TrustedTypePolicyFactory> TrustedTypePolicyFactory::create(ScriptExecutionContext& context)
{
    return adoptRef(*new TrustedTypePolicyFactory(context));
}

TrustedTypePolicyFactory::TrustedTypePolicyFactory(ScriptExecutionContext& context)
    : ContextDestructionObserver(&context)
{ }

ExceptionOr<Ref<TrustedTypePolicy>> TrustedTypePolicyFactory::createPolicy(ScriptExecutionContext& context, const String& policyName, const TrustedTypePolicyOptions& options)
{
    auto csp = context.checkedContentSecurityPolicy();
    ASSERT(csp);

    AllowTrustedTypePolicy policyAllowed = csp->allowTrustedTypesPolicy(policyName, m_createdPolicyNames.contains(policyName));

    switch (policyAllowed) {
    case AllowTrustedTypePolicy::DisallowedName:
        return Exception {
            ExceptionCode::TypeError,
            makeString("Failed to execute 'createPolicy': Policy with name '"_s, policyName, "' disallowed."_s)
        };
    case AllowTrustedTypePolicy::DisallowedDuplicateName:
        return Exception {
            ExceptionCode::TypeError,
            makeString("Failed to execute 'createPolicy': Policy with name '"_s, policyName, "' already exists."_s)
        };
    default:
        auto policy = TrustedTypePolicy::create(policyName, options);
        if (policyName == "default"_s)
            m_defaultPolicy = policy.ptr();

        m_createdPolicyNames.add(policyName);
        return policy;
    }
}

bool TrustedTypePolicyFactory::isHTML(JSC::JSValue value) const
{
    return JSC::jsDynamicCast<JSTrustedHTML*>(value);
}

bool TrustedTypePolicyFactory::isScript(JSC::JSValue value) const
{
    return JSC::jsDynamicCast<JSTrustedScript*>(value);
}

bool TrustedTypePolicyFactory::isScriptURL(JSC::JSValue value) const
{
    return JSC::jsDynamicCast<JSTrustedScriptURL*>(value);
}

Ref<TrustedHTML> TrustedTypePolicyFactory::emptyHTML() const
{
    return TrustedHTML::create(""_s);
}

Ref<TrustedScript> TrustedTypePolicyFactory::emptyScript() const
{
    return TrustedScript::create(""_s);
}

String TrustedTypePolicyFactory::getAttributeType(const String& tagName, const String& attributeParameter, const String& elementNamespace, const String& attributeNamespace) const
{
    return trustedTypeForAttribute(tagName, attributeParameter.convertToASCIILowercase(), elementNamespace, attributeNamespace).attributeType;
}

String TrustedTypePolicyFactory::getPropertyType(const String& tagName, const String& property, const String& elementNamespace) const
{
    auto localName = tagName.convertToASCIILowercase();
    AtomString elementNS = elementNamespace.isEmpty() ? HTMLNames::xhtmlNamespaceURI : AtomString(elementNamespace);

    if (property == "innerHTML"_s || property == "outerHTML"_s)
        return trustedTypeToString(TrustedType::TrustedHTML);

    const QualifiedName element(nullAtom(), AtomString(localName), elementNS);

    if (element.matches(HTMLNames::iframeTag) && property == "srcdoc"_s)
        return trustedTypeToString(TrustedType::TrustedHTML);
    if (element.matches(HTMLNames::scriptTag) && property == "src"_s)
        return trustedTypeToString(TrustedType::TrustedScriptURL);
    if (element.matches(HTMLNames::scriptTag) && (property == "innerText"_s || property == "textContent"_s || property == "text"_s))
        return trustedTypeToString(TrustedType::TrustedScript);

    return nullString();
}

}
