/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 13, 2022.
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
#ifndef __XML_XSLT_FUNCTIONS_H__
#define __XML_XSLT_FUNCTIONS_H__

#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>
#include "xsltexports.h"
#include "xsltInternals.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * XSLT_REGISTER_FUNCTION_LOOKUP:
 *
 * Registering macro, not general purpose at all but used in different modules.
 */
#define XSLT_REGISTER_FUNCTION_LOOKUP(ctxt)			\
    xmlXPathRegisterFuncLookup((ctxt)->xpathCtxt,		\
	xsltXPathFunctionLookup,				\
	(void *)(ctxt->xpathCtxt));

XSLTPUBFUN xmlXPathFunction XSLTCALL
	xsltXPathFunctionLookup		(void *vctxt,
					 const xmlChar *name,
					 const xmlChar *ns_uri);

/*
 * Interfaces for the functions implementations.
 */

XSLTPUBFUN void XSLTCALL
	xsltDocumentFunction		(xmlXPathParserContextPtr ctxt,
					 int nargs);
XSLTPUBFUN void XSLTCALL
	xsltKeyFunction			(xmlXPathParserContextPtr ctxt,
					 int nargs);
XSLTPUBFUN void XSLTCALL
	xsltUnparsedEntityURIFunction	(xmlXPathParserContextPtr ctxt,
					 int nargs);
XSLTPUBFUN void XSLTCALL
	xsltFormatNumberFunction	(xmlXPathParserContextPtr ctxt,
					 int nargs);
XSLTPUBFUN void XSLTCALL
	xsltGenerateIdFunction		(xmlXPathParserContextPtr ctxt,
					 int nargs);
XSLTPUBFUN void XSLTCALL
	xsltSystemPropertyFunction	(xmlXPathParserContextPtr ctxt,
					 int nargs);
XSLTPUBFUN void XSLTCALL
	xsltElementAvailableFunction	(xmlXPathParserContextPtr ctxt,
					 int nargs);
XSLTPUBFUN void XSLTCALL
	xsltFunctionAvailableFunction	(xmlXPathParserContextPtr ctxt,
					 int nargs);

/*
 * And the registration
 */

XSLTPUBFUN void XSLTCALL
	xsltRegisterAllFunctions	(xmlXPathContextPtr ctxt);

#ifdef __cplusplus
}
#endif

#endif /* __XML_XSLT_FUNCTIONS_H__ */

