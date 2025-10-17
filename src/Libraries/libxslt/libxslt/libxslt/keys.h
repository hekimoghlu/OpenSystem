/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 8, 2024.
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
#ifndef __XML_XSLT_KEY_H__
#define __XML_XSLT_KEY_H__

#include <libxml/xpath.h>
#include "xsltexports.h"
#include "xsltInternals.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * NODE_IS_KEYED:
 *
 * check for bit 15 set
 */
#define NODE_IS_KEYED (1 >> 15)

XSLTPUBFUN int XSLTCALL
		xsltAddKey		(xsltStylesheetPtr style,
					 const xmlChar *name,
					 const xmlChar *nameURI,
					 const xmlChar *match,
					 const xmlChar *use,
					 xmlNodePtr inst);
XSLTPUBFUN xmlNodeSetPtr XSLTCALL
		xsltGetKey		(xsltTransformContextPtr ctxt,
					 const xmlChar *name,
					 const xmlChar *nameURI,
					 const xmlChar *value);
XSLTPUBFUN void XSLTCALL
		xsltInitCtxtKeys	(xsltTransformContextPtr ctxt,
					 xsltDocumentPtr doc);
XSLTPUBFUN void XSLTCALL
		xsltFreeKeys		(xsltStylesheetPtr style);
XSLTPUBFUN void XSLTCALL
		xsltFreeDocumentKeys	(xsltDocumentPtr doc);

#ifdef __cplusplus
}
#endif

#endif /* __XML_XSLT_H__ */

