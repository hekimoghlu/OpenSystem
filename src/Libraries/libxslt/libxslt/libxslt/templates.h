/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 15, 2022.
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
#ifndef __XML_XSLT_TEMPLATES_H__
#define __XML_XSLT_TEMPLATES_H__

#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>
#include "xsltexports.h"
#include "xsltInternals.h"

#ifdef __cplusplus
extern "C" {
#endif

XSLTPUBFUN int XSLTCALL
		xsltEvalXPathPredicate		(xsltTransformContextPtr ctxt,
						 xmlXPathCompExprPtr comp,
		                                 xmlNsPtr *nsList,
						 int nsNr);
XSLTPUBFUN xmlChar * XSLTCALL
		xsltEvalTemplateString		(xsltTransformContextPtr ctxt,
						 xmlNodePtr contextNode,
						 xmlNodePtr inst);
XSLTPUBFUN xmlChar * XSLTCALL
		xsltEvalAttrValueTemplate	(xsltTransformContextPtr ctxt,
						 xmlNodePtr node,
						 const xmlChar *name,
						 const xmlChar *ns);
XSLTPUBFUN const xmlChar * XSLTCALL
		xsltEvalStaticAttrValueTemplate	(xsltStylesheetPtr style,
						 xmlNodePtr node,
						 const xmlChar *name,
						 const xmlChar *ns,
						 int *found);

/* TODO: this is obviously broken ... the namespaces should be passed too ! */
XSLTPUBFUN xmlChar * XSLTCALL
		xsltEvalXPathString		(xsltTransformContextPtr ctxt,
						 xmlXPathCompExprPtr comp);
XSLTPUBFUN xmlChar * XSLTCALL
		xsltEvalXPathStringNs		(xsltTransformContextPtr ctxt,
						 xmlXPathCompExprPtr comp,
						 int nsNr,
						 xmlNsPtr *nsList);

XSLTPUBFUN xmlNodePtr * XSLTCALL
		xsltTemplateProcess		(xsltTransformContextPtr ctxt,
						 xmlNodePtr node);
XSLTPUBFUN xmlAttrPtr XSLTCALL
		xsltAttrListTemplateProcess	(xsltTransformContextPtr ctxt,
						 xmlNodePtr target,
						 xmlAttrPtr cur);
XSLTPUBFUN xmlAttrPtr XSLTCALL
		xsltAttrTemplateProcess		(xsltTransformContextPtr ctxt,
						 xmlNodePtr target,
						 xmlAttrPtr attr);
XSLTPUBFUN xmlChar * XSLTCALL
		xsltAttrTemplateValueProcess	(xsltTransformContextPtr ctxt,
						 const xmlChar* attr);
XSLTPUBFUN xmlChar * XSLTCALL
		xsltAttrTemplateValueProcessNode(xsltTransformContextPtr ctxt,
						 const xmlChar* str,
						 xmlNodePtr node);
#ifdef __cplusplus
}
#endif

#endif /* __XML_XSLT_TEMPLATES_H__ */

