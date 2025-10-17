/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 7, 2021.
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
#ifndef __XML_XSLT_NAMESPACES_H__
#define __XML_XSLT_NAMESPACES_H__

#include <libxml/tree.h>
#include "xsltexports.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Used within nsAliases hashtable when the default namespace is required
 * but it's not been explicitly defined
 */
/**
 * UNDEFINED_DEFAULT_NS:
 *
 * Special value for undefined namespace, internal
 */
#define	UNDEFINED_DEFAULT_NS	(const xmlChar *) -1L

XSLTPUBFUN void XSLTCALL
		xsltNamespaceAlias	(xsltStylesheetPtr style,
					 xmlNodePtr node);
XSLTPUBFUN xmlNsPtr XSLTCALL
		xsltGetNamespace	(xsltTransformContextPtr ctxt,
					 xmlNodePtr cur,
					 xmlNsPtr ns,
					 xmlNodePtr out);
XSLTPUBFUN xmlNsPtr XSLTCALL
		xsltGetPlainNamespace	(xsltTransformContextPtr ctxt,
					 xmlNodePtr cur,
					 xmlNsPtr ns,
					 xmlNodePtr out);
XSLTPUBFUN xmlNsPtr XSLTCALL
		xsltGetSpecialNamespace	(xsltTransformContextPtr ctxt,
					 xmlNodePtr cur,
					 const xmlChar *URI,
					 const xmlChar *prefix,
					 xmlNodePtr out);
XSLTPUBFUN xmlNsPtr XSLTCALL
		xsltCopyNamespace	(xsltTransformContextPtr ctxt,
					 xmlNodePtr elem,
					 xmlNsPtr ns);
XSLTPUBFUN xmlNsPtr XSLTCALL
		xsltCopyNamespaceList	(xsltTransformContextPtr ctxt,
					 xmlNodePtr node,
					 xmlNsPtr cur);
XSLTPUBFUN void XSLTCALL
		xsltFreeNamespaceAliasHashes
					(xsltStylesheetPtr style);

#ifdef __cplusplus
}
#endif

#endif /* __XML_XSLT_NAMESPACES_H__ */

