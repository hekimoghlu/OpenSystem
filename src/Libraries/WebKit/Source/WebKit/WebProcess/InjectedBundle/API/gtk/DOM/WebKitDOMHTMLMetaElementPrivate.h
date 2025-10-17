/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 21, 2025.
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
#ifndef WebKitDOMHTMLMetaElementPrivate_h
#define WebKitDOMHTMLMetaElementPrivate_h

#include <WebCore/HTMLMetaElement.h>
#include <webkitdom/WebKitDOMHTMLMetaElement.h>

namespace WebKit {
WebKitDOMHTMLMetaElement* wrapHTMLMetaElement(WebCore::HTMLMetaElement*);
WebKitDOMHTMLMetaElement* kit(WebCore::HTMLMetaElement*);
WebCore::HTMLMetaElement* core(WebKitDOMHTMLMetaElement*);
} // namespace WebKit

#endif /* WebKitDOMHTMLMetaElementPrivate_h */
