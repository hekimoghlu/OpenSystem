/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 11, 2022.
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
#include "JSBasicCredential.h"

#if ENABLE(WEB_AUTHN)

#include "JSDOMBinding.h"
#include "JSDigitalCredential.h"
#include "JSPublicKeyCredential.h"

namespace WebCore {
using namespace JSC;

JSValue toJSNewlyCreated(JSC::JSGlobalObject*, JSDOMGlobalObject* globalObject, Ref<BasicCredential>&& credential)
{
    if (is<PublicKeyCredential>(credential))
        return createWrapper<PublicKeyCredential>(globalObject, WTFMove(credential));
    if (is<DigitalCredential>(credential))
        return createWrapper<DigitalCredential>(globalObject, WTFMove(credential));
    return createWrapper<BasicCredential>(globalObject, WTFMove(credential));
}

JSValue toJS(JSGlobalObject* lexicalGlobalObject, JSDOMGlobalObject* globalObject, BasicCredential& credential)
{
    return wrap(lexicalGlobalObject, globalObject, credential);
}

} // namespace WebCore

#endif // ENABLE(WEB_AUTHN)
