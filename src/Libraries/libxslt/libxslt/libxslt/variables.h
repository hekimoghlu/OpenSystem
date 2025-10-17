/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 10, 2024.
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
#ifndef __XML_XSLT_VARIABLES_H__
#define __XML_XSLT_VARIABLES_H__

#include <libxml/xpath.h>
#include <libxml/xpathInternals.h>
#include "xsltexports.h"
#include "xsltInternals.h"
#include "functions.h"

#ifdef __cplusplus
extern "C" {
#endif


/**
 * XSLT_REGISTER_VARIABLE_LOOKUP:
 *
 * Registering macro, not general purpose at all but used in different modules.
 */

#define XSLT_REGISTER_VARIABLE_LOOKUP(ctxt)			\
    xmlXPathRegisterVariableLookup((ctxt)->xpathCtxt,		\
	       xsltXPathVariableLookup,	(void *)(ctxt));	\
    xsltRegisterAllFunctions((ctxt)->xpathCtxt);		\
    xsltRegisterAllElement(ctxt);				\
    (ctxt)->xpathCtxt->extra = ctxt

/*
 * Flags for memory management of RVTs
 */

/**
 * XSLT_RVT_LOCAL:
 *
 * RVT is destroyed after the current instructions ends.
 */
#ifdef LIBXSLT_API_FOR_MACOS14_IOS17_WATCHOS10_TVOS17
#define XSLT_RVT_LOCAL       1
#else
#define XSLT_RVT_LOCAL       ((void *)1)
#endif

/**
 * XSLT_RVT_FUNC_RESULT:
 *
 * RVT is part of results returned with func:result. The RVT won't be
 * destroyed after exiting a template and will be reset to XSLT_RVT_LOCAL or
 * XSLT_RVT_VARIABLE in the template that receives the return value.
 */
#ifdef LIBXSLT_API_FOR_MACOS14_IOS17_WATCHOS10_TVOS17
#define XSLT_RVT_FUNC_RESULT 2
#else
#define XSLT_RVT_FUNC_RESULT ((void *)2)
#endif

/**
 * XSLT_RVT_GLOBAL:
 *
 * RVT is part of a global variable.
 */
#ifdef LIBXSLT_API_FOR_MACOS14_IOS17_WATCHOS10_TVOS17
#define XSLT_RVT_GLOBAL      3
#else
#define XSLT_RVT_GLOBAL      ((void *)3)
#endif

/*
 * Interfaces for the variable module.
 */

XSLTPUBFUN int XSLTCALL
		xsltEvalGlobalVariables		(xsltTransformContextPtr ctxt);
XSLTPUBFUN int XSLTCALL
		xsltEvalUserParams		(xsltTransformContextPtr ctxt,
						 const char **params);
XSLTPUBFUN int XSLTCALL
		xsltQuoteUserParams		(xsltTransformContextPtr ctxt,
						 const char **params);
XSLTPUBFUN int XSLTCALL
		xsltEvalOneUserParam		(xsltTransformContextPtr ctxt,
						 const xmlChar * name,
						 const xmlChar * value);
XSLTPUBFUN int XSLTCALL
		xsltQuoteOneUserParam		(xsltTransformContextPtr ctxt,
						 const xmlChar * name,
						 const xmlChar * value);

XSLTPUBFUN void XSLTCALL
		xsltParseGlobalVariable		(xsltStylesheetPtr style,
						 xmlNodePtr cur);
XSLTPUBFUN void XSLTCALL
		xsltParseGlobalParam		(xsltStylesheetPtr style,
						 xmlNodePtr cur);
XSLTPUBFUN void XSLTCALL
		xsltParseStylesheetVariable	(xsltTransformContextPtr ctxt,
						 xmlNodePtr cur);
XSLTPUBFUN void XSLTCALL
		xsltParseStylesheetParam	(xsltTransformContextPtr ctxt,
						 xmlNodePtr cur);
XSLTPUBFUN xsltStackElemPtr XSLTCALL
		xsltParseStylesheetCallerParam	(xsltTransformContextPtr ctxt,
						 xmlNodePtr cur);
XSLTPUBFUN int XSLTCALL
		xsltAddStackElemList		(xsltTransformContextPtr ctxt,
						 xsltStackElemPtr elems);
XSLTPUBFUN void XSLTCALL
		xsltFreeGlobalVariables		(xsltTransformContextPtr ctxt);
XSLTPUBFUN xmlXPathObjectPtr XSLTCALL
		xsltVariableLookup		(xsltTransformContextPtr ctxt,
						 const xmlChar *name,
						 const xmlChar *ns_uri);
XSLTPUBFUN xmlXPathObjectPtr XSLTCALL
		xsltXPathVariableLookup		(void *ctxt,
						 const xmlChar *name,
						 const xmlChar *ns_uri);
#ifdef __cplusplus
}
#endif

#endif /* __XML_XSLT_VARIABLES_H__ */

