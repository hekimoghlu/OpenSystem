/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 19, 2023.
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
#ifndef WKAPICastWPE_h
#define WKAPICastWPE_h

#ifndef WKAPICast_h
#error "Please #include \"WKAPICast.h\" instead of this file directly."
#endif

#include "WKView.h"
#include "WebGrammarDetail.h"

namespace WKWPE {
class View;
}

namespace WebKit {

WK_ADD_API_MAPPING(WKGrammarDetailRef, WebGrammarDetail)
WK_ADD_API_MAPPING(WKViewRef, WKWPE::View)

inline ProxyingRefPtr<WebGrammarDetail> toAPI(const WebCore::GrammarDetail& grammarDetail)
{
    return ProxyingRefPtr<WebGrammarDetail>(WebGrammarDetail::create(grammarDetail));
}

}

#endif // WKAPICastWPE_h
