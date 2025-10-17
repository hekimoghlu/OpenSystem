/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 22, 2022.
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

#include "ExceptionOr.h"
#include "URLDecomposition.h"
#include <wtf/RefCountedAndCanMakeWeakPtr.h>
#include <wtf/URL.h>
#include <wtf/WeakPtr.h>

namespace WebCore {

class Blob;
class ScriptExecutionContext;
class URLRegistrable;
class URLSearchParams;

class DOMURL final : public RefCountedAndCanMakeWeakPtr<DOMURL>, public URLDecomposition {
public:
    static ExceptionOr<Ref<DOMURL>> create(const String& url, const String& base);
    WEBCORE_EXPORT ~DOMURL();

    static RefPtr<DOMURL> parse(const String& url, const String& base);
    static bool canParse(const String& url, const String& base);

    const URL& href() const { return m_url; }
    ExceptionOr<void> setHref(const String&);

    URLSearchParams& searchParams();

    const String& toJSON() const { return m_url.string(); }

    static String createObjectURL(ScriptExecutionContext&, Blob&);
    static void revokeObjectURL(ScriptExecutionContext&, const String&);

    static String createPublicURL(ScriptExecutionContext&, URLRegistrable&);

private:
    static ExceptionOr<Ref<DOMURL>> create(const String& url, const URL& base);
    DOMURL(URL&& completeURL);

    URL fullURL() const final { return m_url; }
    void setFullURL(const URL& fullURL) final { setHref(fullURL.string()); }

    URL m_url;
    RefPtr<URLSearchParams> m_searchParams;
};

} // namespace WebCore
