/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 25, 2024.
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
#ifndef __XML_XSLT_IMPORTS_H__
#define __XML_XSLT_IMPORTS_H__

#include <libxml/tree.h>
#include "xsltexports.h"
#include "xsltInternals.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * XSLT_GET_IMPORT_PTR:
 *
 * A macro to import pointers from the stylesheet cascading order.
 */
#define XSLT_GET_IMPORT_PTR(res, style, name) {			\
    xsltStylesheetPtr st = style;				\
    res = NULL;							\
    while (st != NULL) {					\
	if (st->name != NULL) { res = st->name; break; }	\
	st = xsltNextImport(st);				\
    }}

/**
 * XSLT_GET_IMPORT_INT:
 *
 * A macro to import intergers from the stylesheet cascading order.
 */
#define XSLT_GET_IMPORT_INT(res, style, name) {			\
    xsltStylesheetPtr st = style;				\
    res = -1;							\
    while (st != NULL) {					\
	if (st->name != -1) { res = st->name; break; }	\
	st = xsltNextImport(st);				\
    }}

/*
 * Module interfaces
 */
XSLTPUBFUN int XSLTCALL
			xsltParseStylesheetImport(xsltStylesheetPtr style,
						  xmlNodePtr cur);
XSLTPUBFUN int XSLTCALL
			xsltParseStylesheetInclude
						 (xsltStylesheetPtr style,
						  xmlNodePtr cur);
XSLTPUBFUN xsltStylesheetPtr XSLTCALL
			xsltNextImport		 (xsltStylesheetPtr style);
XSLTPUBFUN int XSLTCALL
			xsltNeedElemSpaceHandling(xsltTransformContextPtr ctxt);
XSLTPUBFUN int XSLTCALL
			xsltFindElemSpaceHandling(xsltTransformContextPtr ctxt,
						  xmlNodePtr node);
XSLTPUBFUN xsltTemplatePtr XSLTCALL
			xsltFindTemplate	 (xsltTransformContextPtr ctxt,
						  const xmlChar *name,
						  const xmlChar *nameURI);

#ifdef __cplusplus
}
#endif

#endif /* __XML_XSLT_IMPORTS_H__ */

