/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 29, 2025.
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
#include "ContentSecurityPolicyDirectiveNames.h"

#include <wtf/text/ASCIILiteral.h>

namespace WebCore {

namespace ContentSecurityPolicyDirectiveNames {

ASCIILiteral baseURI = "base-uri"_s;
ASCIILiteral childSrc = "child-src"_s;
ASCIILiteral connectSrc = "connect-src"_s;
ASCIILiteral defaultSrc = "default-src"_s;
ASCIILiteral fontSrc = "font-src"_s;
ASCIILiteral formAction = "form-action"_s;
ASCIILiteral frameAncestors = "frame-ancestors"_s;
ASCIILiteral frameSrc = "frame-src"_s;
#if ENABLE(APPLICATION_MANIFEST)
ASCIILiteral manifestSrc = "manifest-src"_s;
#endif
ASCIILiteral imgSrc = "img-src"_s;
ASCIILiteral mediaSrc = "media-src"_s;
ASCIILiteral objectSrc = "object-src"_s;
ASCIILiteral pluginTypes = "plugin-types"_s;
ASCIILiteral prefetchSrc = "prefetch-src"_s;
ASCIILiteral reportTo = "report-to"_s;
ASCIILiteral reportURI = "report-uri"_s;
ASCIILiteral requireTrustedTypesFor = "require-trusted-types-for"_s;
ASCIILiteral sandbox = "sandbox"_s;
ASCIILiteral scriptSrc = "script-src"_s;
ASCIILiteral scriptSrcAttr = "script-src-attr"_s;
ASCIILiteral scriptSrcElem = "script-src-elem"_s;
ASCIILiteral styleSrc = "style-src"_s;
ASCIILiteral styleSrcAttr = "style-src-attr"_s;
ASCIILiteral styleSrcElem = "style-src-elem"_s;
ASCIILiteral trustedTypes = "trusted-types"_s;
ASCIILiteral upgradeInsecureRequests = "upgrade-insecure-requests"_s;
ASCIILiteral blockAllMixedContent = "block-all-mixed-content"_s;
ASCIILiteral workerSrc = "worker-src"_s;

} // namespace ContentSecurityPolicyDirectiveNames

} // namespace WebCore
