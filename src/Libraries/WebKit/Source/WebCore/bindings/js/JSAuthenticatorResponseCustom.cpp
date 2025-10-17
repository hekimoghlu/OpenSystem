/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 4, 2025.
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
#include "JSAuthenticatorResponse.h"

#if ENABLE(WEB_AUTHN)

#include "JSAuthenticatorAssertionResponse.h"
#include "JSAuthenticatorAttestationResponse.h"
#include "JSDOMBinding.h"

namespace WebCore {
using namespace JSC;

JSValue toJSNewlyCreated(JSC::JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<AuthenticatorResponse>&& response)
{
    if (is<AuthenticatorAttestationResponse>(response))
        return createWrapper<AuthenticatorAttestationResponse>(globalObject, WTFMove(response));
    if (is<AuthenticatorAssertionResponse>(response))
        return createWrapper<AuthenticatorAssertionResponse>(globalObject, WTFMove(response));
    return createWrapper<AuthenticatorResponse>(globalObject, WTFMove(response));
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, AuthenticatorResponse& response)
{
    return wrap(lexicalGlobalObject, globalObject, response);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
