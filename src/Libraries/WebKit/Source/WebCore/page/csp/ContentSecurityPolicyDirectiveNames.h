/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 7, 2025.
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

#include <wtf/Forward.h>

namespace WebCore {

namespace ContentSecurityPolicyDirectiveNames {

extern ASCIILiteral baseURI;
extern ASCIILiteral childSrc;
extern ASCIILiteral connectSrc;
extern ASCIILiteral defaultSrc;
extern ASCIILiteral fontSrc;
extern ASCIILiteral formAction;
extern ASCIILiteral frameAncestors;
extern ASCIILiteral frameSrc;
extern ASCIILiteral imgSrc;
#if ENABLE(APPLICATION_MANIFEST)
extern ASCIILiteral manifestSrc;
#endif
extern ASCIILiteral mediaSrc;
extern ASCIILiteral objectSrc;
extern ASCIILiteral pluginTypes;
extern ASCIILiteral prefetchSrc;
extern ASCIILiteral reportURI;
extern ASCIILiteral reportTo;
extern ASCIILiteral requireTrustedTypesFor;
extern ASCIILiteral sandbox;
extern ASCIILiteral scriptSrc;
extern ASCIILiteral scriptSrcElem;
extern ASCIILiteral scriptSrcAttr;
extern ASCIILiteral styleSrc;
extern ASCIILiteral styleSrcAttr;
extern ASCIILiteral styleSrcElem;
extern ASCIILiteral trustedTypes;
extern ASCIILiteral upgradeInsecureRequests;
extern ASCIILiteral blockAllMixedContent;
extern ASCIILiteral workerSrc;

} // namespace ContentSecurityPolicyDirectiveNames

} // namespace WebCore

