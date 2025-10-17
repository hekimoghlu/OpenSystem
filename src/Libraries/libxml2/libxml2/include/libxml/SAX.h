/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 12, 2022.
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
#ifndef __XML_SAX_H__
#define __XML_SAX_H__

#include <stdio.h>
#include <stdlib.h>
#include <libxml/xmlversion.h>
#include <libxml/parser.h>
#include <libxml/xlink.h>

#ifdef LIBXML_LEGACY_ENABLED

#ifdef __cplusplus
extern "C" {
#endif
XMLPUBFUN const xmlChar * XMLCALL
		getPublicId			(void *ctx);
XMLPUBFUN const xmlChar * XMLCALL
		getSystemId			(void *ctx);
XMLPUBFUN void XMLCALL
		setDocumentLocator		(void *ctx,
						 xmlSAXLocatorPtr loc);

XMLPUBFUN int XMLCALL
		getLineNumber			(void *ctx);
XMLPUBFUN int XMLCALL
		getColumnNumber			(void *ctx);

XMLPUBFUN int XMLCALL
		isStandalone			(void *ctx);
XMLPUBFUN int XMLCALL
		hasInternalSubset		(void *ctx);
XMLPUBFUN int XMLCALL
		hasExternalSubset		(void *ctx);

XMLPUBFUN void XMLCALL
		internalSubset			(void *ctx,
						 const xmlChar *name,
						 const xmlChar *ExternalID,
						 const xmlChar *SystemID);
XMLPUBFUN void XMLCALL
		externalSubset			(void *ctx,
						 const xmlChar *name,
						 const xmlChar *ExternalID,
						 const xmlChar *SystemID);
XMLPUBFUN xmlEntityPtr XMLCALL
		getEntity			(void *ctx,
						 const xmlChar *name);
XMLPUBFUN xmlEntityPtr XMLCALL
		getParameterEntity		(void *ctx,
						 const xmlChar *name);
XMLPUBFUN xmlParserInputPtr XMLCALL
		resolveEntity			(void *ctx,
						 const xmlChar *publicId,
						 const xmlChar *systemId);

XMLPUBFUN void XMLCALL
		entityDecl			(void *ctx,
						 const xmlChar *name,
						 int type,
						 const xmlChar *publicId,
						 const xmlChar *systemId,
						 xmlChar *content);
XMLPUBFUN void XMLCALL
		attributeDecl			(void *ctx,
						 const xmlChar *elem,
						 const xmlChar *fullname,
						 int type,
						 int def,
						 const xmlChar *defaultValue,
						 xmlEnumerationPtr tree);
XMLPUBFUN void XMLCALL
		elementDecl			(void *ctx,
						 const xmlChar *name,
						 int type,
						 xmlElementContentPtr content);
XMLPUBFUN void XMLCALL
		notationDecl			(void *ctx,
						 const xmlChar *name,
						 const xmlChar *publicId,
						 const xmlChar *systemId);
XMLPUBFUN void XMLCALL
		unparsedEntityDecl		(void *ctx,
						 const xmlChar *name,
						 const xmlChar *publicId,
						 const xmlChar *systemId,
						 const xmlChar *notationName);

XMLPUBFUN void XMLCALL
		startDocument			(void *ctx);
XMLPUBFUN void XMLCALL
		endDocument			(void *ctx);
XMLPUBFUN void XMLCALL
		attribute			(void *ctx,
						 const xmlChar *fullname,
						 const xmlChar *value);
XMLPUBFUN void XMLCALL
		startElement			(void *ctx,
						 const xmlChar *fullname,
						 const xmlChar **atts);
XMLPUBFUN void XMLCALL
		endElement			(void *ctx,
						 const xmlChar *name);
XMLPUBFUN void XMLCALL
		reference			(void *ctx,
						 const xmlChar *name);
XMLPUBFUN void XMLCALL
		characters			(void *ctx,
						 const xmlChar *ch,
						 int len);
XMLPUBFUN void XMLCALL
		ignorableWhitespace		(void *ctx,
						 const xmlChar *ch,
						 int len);
XMLPUBFUN void XMLCALL
		processingInstruction		(void *ctx,
						 const xmlChar *target,
						 const xmlChar *data);
XMLPUBFUN void XMLCALL
		globalNamespace			(void *ctx,
						 const xmlChar *href,
						 const xmlChar *prefix);
XMLPUBFUN void XMLCALL
		setNamespace			(void *ctx,
						 const xmlChar *name);
XMLPUBFUN xmlNsPtr XMLCALL
		getNamespace			(void *ctx);
XMLPUBFUN int XMLCALL
		checkNamespace			(void *ctx,
						 xmlChar *nameSpace);
XMLPUBFUN void XMLCALL
		namespaceDecl			(void *ctx,
						 const xmlChar *href,
						 const xmlChar *prefix);
XMLPUBFUN void XMLCALL
		comment				(void *ctx,
						 const xmlChar *value);
XMLPUBFUN void XMLCALL
		cdataBlock			(void *ctx,
						 const xmlChar *value,
						 int len);

#ifdef LIBXML_SAX1_ENABLED
XMLPUBFUN void XMLCALL
		initxmlDefaultSAXHandler	(xmlSAXHandlerV1 *hdlr,
						 int warning);
#ifdef LIBXML_HTML_ENABLED
XMLPUBFUN void XMLCALL
		inithtmlDefaultSAXHandler	(xmlSAXHandlerV1 *hdlr);
#endif
#ifdef LIBXML_DOCB_ENABLED
XMLPUBFUN void XMLCALL
		initdocbDefaultSAXHandler	(xmlSAXHandlerV1 *hdlr);
#endif
#endif /* LIBXML_SAX1_ENABLED */

#ifdef __cplusplus
}
#endif

#endif /* LIBXML_LEGACY_ENABLED */

#endif /* __XML_SAX_H__ */
