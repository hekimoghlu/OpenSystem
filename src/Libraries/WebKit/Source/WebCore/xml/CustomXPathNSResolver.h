/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 23, 2025.
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

#include "ActiveDOMCallback.h"
#include "CallbackResult.h"
#include "XPathNSResolver.h"
#include <wtf/text/WTFString.h>

namespace WebCore {

class CustomXPathNSResolver : public XPathNSResolver, public ActiveDOMCallback {
public:
    using ActiveDOMCallback::ActiveDOMCallback;

    virtual CallbackResult<String> lookupNamespaceURIForBindings(const AtomString& prefix) = 0;
    virtual CallbackResult<String> lookupNamespaceURIForBindingsRethrowingException(const AtomString& prefix) = 0;

    AtomString lookupNamespaceURI(const AtomString& prefix);

private:
    virtual bool hasCallback() const = 0;
};

} // namespace WebCore
