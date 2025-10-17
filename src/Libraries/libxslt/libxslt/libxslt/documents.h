/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, September 23, 2022.
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
#ifndef __XML_XSLT_DOCUMENTS_H__
#define __XML_XSLT_DOCUMENTS_H__

#include <libxml/tree.h>
#include "xsltexports.h"
#include "xsltInternals.h"

#ifdef __cplusplus
extern "C" {
#endif

XSLTPUBFUN xsltDocumentPtr XSLTCALL
		xsltNewDocument		(xsltTransformContextPtr ctxt,
					 xmlDocPtr doc);
XSLTPUBFUN xsltDocumentPtr XSLTCALL
		xsltLoadDocument	(xsltTransformContextPtr ctxt,
					 const xmlChar *URI);
XSLTPUBFUN xsltDocumentPtr XSLTCALL
		xsltFindDocument	(xsltTransformContextPtr ctxt,
					 xmlDocPtr doc);
XSLTPUBFUN void XSLTCALL
		xsltFreeDocuments	(xsltTransformContextPtr ctxt);

XSLTPUBFUN xsltDocumentPtr XSLTCALL
		xsltLoadStyleDocument	(xsltStylesheetPtr style,
					 const xmlChar *URI);
XSLTPUBFUN xsltDocumentPtr XSLTCALL
		xsltNewStyleDocument	(xsltStylesheetPtr style,
					 xmlDocPtr doc);
XSLTPUBFUN void XSLTCALL
		xsltFreeStyleDocuments	(xsltStylesheetPtr style);

/*
 * Hooks for document loading
 */

/**
 * xsltLoadType:
 *
 * Enum defining the kind of loader requirement.
 */
typedef enum {
    XSLT_LOAD_START = 0,	/* loading for a top stylesheet */
    XSLT_LOAD_STYLESHEET = 1,	/* loading for a stylesheet include/import */
    XSLT_LOAD_DOCUMENT = 2	/* loading document at transformation time */
} xsltLoadType;

/**
 * xsltDocLoaderFunc:
 * @URI: the URI of the document to load
 * @dict: the dictionary to use when parsing that document
 * @options: parsing options, a set of xmlParserOption
 * @ctxt: the context, either a stylesheet or a transformation context
 * @type: the xsltLoadType indicating the kind of loading required
 *
 * An xsltDocLoaderFunc is a signature for a function which can be
 * registered to load document not provided by the compilation or
 * transformation API themselve, for example when an xsl:import,
 * xsl:include is found at compilation time or when a document()
 * call is made at runtime.
 *
 * Returns the pointer to the document (which will be modified and
 * freed by the engine later), or NULL in case of error.
 */
typedef xmlDocPtr (*xsltDocLoaderFunc)		(const xmlChar *URI,
						 xmlDictPtr dict,
						 int options,
						 void *ctxt,
						 xsltLoadType type);

XSLTPUBFUN void XSLTCALL
		xsltSetLoaderFunc		(xsltDocLoaderFunc f);

/* the loader may be needed by extension libraries so it is exported */
XSLTPUBVAR xsltDocLoaderFunc xsltDocDefaultLoader;

#ifdef __cplusplus
}
#endif

#endif /* __XML_XSLT_DOCUMENTS_H__ */

