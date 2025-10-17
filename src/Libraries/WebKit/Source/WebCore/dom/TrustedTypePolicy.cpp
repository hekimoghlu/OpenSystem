/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 3, 2024.
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
#include "TrustedTypePolicy.h"

#include "TrustedHTML.h"
#include "TrustedScript.h"
#include "TrustedScriptURL.h"
#include "TrustedType.h"
#include "TrustedTypePolicyOptions.h"
#include "WebCoreOpaqueRoot.h"
#include <wtf/TZoneMallocInlines.h>
#include <wtf/text/MakeString.h>

namespace WebCore {

WTF_MAKE_TZONE_OR_ISO_ALLOCATED_IMPL(TrustedTypePolicy);

Ref<TrustedTypePolicy> TrustedTypePolicy::create(const String& name, const TrustedTypePolicyOptions& options)
{
    return adoptRef(*new TrustedTypePolicy(name, options));
}

TrustedTypePolicy::TrustedTypePolicy(const String& name, const TrustedTypePolicyOptions& options)
    : m_name(name)
    , m_options(options)
{ }

ExceptionOr<Ref<TrustedHTML>> TrustedTypePolicy::createHTML(const String& input, FixedVector<JSC::Strong<JSC::Unknown>>&& arguments)
{
    auto policyValue = getPolicyValue(TrustedType::TrustedHTML, input, WTFMove(arguments));

    if (policyValue.hasException())
        return policyValue.releaseException();

    return TrustedHTML::create(policyValue.releaseReturnValue());
}

ExceptionOr<Ref<TrustedScript>> TrustedTypePolicy::createScript(const String& input, FixedVector<JSC::Strong<JSC::Unknown>>&& arguments)
{
    auto policyValue = getPolicyValue(TrustedType::TrustedScript, input, WTFMove(arguments));

    if (policyValue.hasException())
        return policyValue.releaseException();

    return TrustedScript::create(policyValue.releaseReturnValue());
}

ExceptionOr<Ref<TrustedScriptURL>> TrustedTypePolicy::createScriptURL(const String& input, FixedVector<JSC::Strong<JSC::Unknown>>&& arguments)
{
    auto policyValue = getPolicyValue(TrustedType::TrustedScriptURL, input, WTFMove(arguments));

    if (policyValue.hasException())
        return policyValue.releaseException();

    return TrustedScriptURL::create(policyValue.releaseReturnValue());
}

// https://w3c.github.io/trusted-types/dist/spec/#get-trusted-type-policy-value-algorithm
ExceptionOr<String> TrustedTypePolicy::getPolicyValue(TrustedType trustedTypeName, const String& input, FixedVector<JSC::Strong<JSC::Unknown>>&& arguments, IfMissing ifMissing)
{
    CallbackResult<String> policyValue(CallbackResultType::UnableToExecute);
    if (trustedTypeName == TrustedType::TrustedHTML) {
        RefPtr<CreateHTMLCallback> protectedCreateHTML;
        {
            Locker locker { lock() };
            protectedCreateHTML = m_options.createHTML;
        }
        if (protectedCreateHTML && protectedCreateHTML->hasCallback())
            policyValue = protectedCreateHTML->handleEventRethrowingException(input, WTFMove(arguments));
    } else if (trustedTypeName == TrustedType::TrustedScript) {
        RefPtr<CreateScriptCallback> protectedCreateScript;
        {
            Locker locker { lock() };
            protectedCreateScript = m_options.createScript;
        }
        if (protectedCreateScript && protectedCreateScript->hasCallback())
            policyValue = protectedCreateScript->handleEventRethrowingException(input, WTFMove(arguments));
    } else if (trustedTypeName == TrustedType::TrustedScriptURL) {
        RefPtr<CreateScriptURLCallback> protectedCreateScriptURL;
        {
            Locker locker { lock() };
            protectedCreateScriptURL = m_options.createScriptURL;
        }
        if (protectedCreateScriptURL && protectedCreateScriptURL->hasCallback())
            policyValue = protectedCreateScriptURL->handleEventRethrowingException(input, WTFMove(arguments));
    } else {
        ASSERT_NOT_REACHED();
        return Exception { ExceptionCode::TypeError };
    }

    if (policyValue.type() == CallbackResultType::Success)
        return policyValue.releaseReturnValue();
    if (policyValue.type() == CallbackResultType::ExceptionThrown)
        return Exception { ExceptionCode::ExistingExceptionError };

    if (ifMissing == IfMissing::Throw) {
        return Exception {
            ExceptionCode::TypeError,
            makeString("Policy "_s, m_name,
                "'s TrustedTypePolicyOptions did not specify a '"_s, trustedTypeToCallbackName(trustedTypeName), "' member."_s)
        };
    }

    return String(nullString());
}

WebCoreOpaqueRoot root(TrustedTypePolicy* policy)
{
    return WebCoreOpaqueRoot { policy };
}

} // namespace WebCore
