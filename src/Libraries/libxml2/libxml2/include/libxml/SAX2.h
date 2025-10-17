/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 3, 2025.
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
#ifndef __XML_SAX2_H__
#define __XML_SAX2_H__

#include <stdio.h>
#include <stdlib.h>
#include <libxml/xmlversion.h>
#include <libxml/parser.h>
#include <libxml/xlink.h>

#ifdef __cplusplus
extern "C" {
#endif
XMLPUBFUN const xmlChar * XMLCALL
		xmlSAX2GetPublicId		(void *ctx);
XMLPUBFUN const xmlChar * XMLCALL
		xmlSAX2GetSystemId		(void *ctx);
XMLPUBFUN void XMLCALL
		xmlSAX2SetDocumentLocator	(void *ctx,
						 xmlSAXLocatorPtr loc);

XMLPUBFUN int XMLCALL
		xmlSAX2GetLineNumber		(void *ctx);
XMLPUBFUN int XMLCALL
		xmlSAX2GetColumnNumber		(void *ctx);

XMLPUBFUN int XMLCALL
		xmlSAX2IsStandalone		(void *ctx);
XMLPUBFUN int XMLCALL
		xmlSAX2HasInternalSubset	(void *ctx);
XMLPUBFUN int XMLCALL
		xmlSAX2HasExternalSubset	(void *ctx);

XMLPUBFUN void XMLCALL
		xmlSAX2InternalSubset		(void *ctx,
						 const xmlChar *name,
						 const xmlChar *ExternalID,
						 const xmlChar *SystemID);
XMLPUBFUN void XMLCALL
		xmlSAX2ExternalSubset		(void *ctx,
						 const xmlChar *name,
						 const xmlChar *ExternalID,
						 const xmlChar *SystemID);
XMLPUBFUN xmlEntityPtr XMLCALL
		xmlSAX2GetEntity		(void *ctx,
						 const xmlChar *name);
XMLPUBFUN xmlEntityPtr XMLCALL
		xmlSAX2GetParameterEntity	(void *ctx,
						 const xmlChar *name);
XMLPUBFUN xmlParserInputPtr XMLCALL
		xmlSAX2ResolveEntity		(void *ctx,
						 const xmlChar *publicId,
						 const xmlChar *systemId);

XMLPUBFUN void XMLCALL
		xmlSAX2EntityDecl		(void *ctx,
						 const xmlChar *name,
						 int type,
						 const xmlChar *publicId,
						 const xmlChar *systemId,
						 xmlChar *content);
XMLPUBFUN void XMLCALL
		xmlSAX2AttributeDecl		(void *ctx,
						 const xmlChar *elem,
						 const xmlChar *fullname,
						 int type,
						 int def,
						 const xmlChar *defaultValue,
						 xmlEnumerationPtr tree);
XMLPUBFUN void XMLCALL
		xmlSAX2ElementDecl		(void *ctx,
						 const xmlChar *name,
						 int type,
						 xmlElementContentPtr content);
XMLPUBFUN void XMLCALL
		xmlSAX2NotationDecl		(void *ctx,
						 const xmlChar *name,
						 const xmlChar *publicId,
						 const xmlChar *systemId);
XMLPUBFUN void XMLCALL
		xmlSAX2UnparsedEntityDecl	(void *ctx,
						 const xmlChar *name,
						 const xmlChar *publicId,
						 const xmlChar *systemId,
						 const xmlChar *notationName);

XMLPUBFUN void XMLCALL
		xmlSAX2StartDocument		(void *ctx);
XMLPUBFUN void XMLCALL
		xmlSAX2EndDocument		(void *ctx);
#if defined(LIBXML_SAX1_ENABLED) || defined(LIBXML_HTML_ENABLED) || \
    defined(LIBXML_WRITER_ENABLED) || defined(LIBXML_DOCB_ENABLED) || \
    defined(LIBXML_LEGACY_ENABLED)
XMLPUBFUN void XMLCALL
		xmlSAX2StartElement		(void *ctx,
						 const xmlChar *fullname,
						 const xmlChar **atts);
XMLPUBFUN void XMLCALL
		xmlSAX2EndElement		(void *ctx,
						 const xmlChar *name);
#endif /* LIBXML_SAX1_ENABLED or LIBXML_HTML_ENABLED or LIBXML_LEGACY_ENABLED */
XMLPUBFUN void XMLCALL
		xmlSAX2StartElementNs		(void *ctx,
						 const xmlChar *localname,
						 const xmlChar *prefix,
						 const xmlChar *URI,
						 int nb_namespaces,
						 const xmlChar **namespaces,
						 int nb_attributes,
						 int nb_defaulted,
						 const xmlChar **attributes);
XMLPUBFUN void XMLCALL
		xmlSAX2EndElementNs		(void *ctx,
						 const xmlChar *localname,
						 const xmlChar *prefix,
						 const xmlChar *URI);
XMLPUBFUN void XMLCALL
		xmlSAX2Reference		(void *ctx,
						 const xmlChar *name);
XMLPUBFUN void XMLCALL
		xmlSAX2Characters		(void *ctx,
						 const xmlChar *ch,
						 int len);
XMLPUBFUN void XMLCALL
		xmlSAX2IgnorableWhitespace	(void *ctx,
						 const xmlChar *ch,
						 int len);
XMLPUBFUN void XMLCALL
		xmlSAX2ProcessingInstruction	(void *ctx,
						 const xmlChar *target,
						 const xmlChar *data);
XMLPUBFUN void XMLCALL
		xmlSAX2Comment			(void *ctx,
						 const xmlChar *value);
XMLPUBFUN void XMLCALL
		xmlSAX2CDataBlock		(void *ctx,
						 const xmlChar *value,
						 int len);

#ifdef LIBXML_SAX1_ENABLED
XMLPUBFUN int XMLCALL
		xmlSAXDefaultVersion		(int version);
#endif /* LIBXML_SAX1_ENABLED */

XMLPUBFUN int XMLCALL
		xmlSAXVersion			(xmlSAXHandler *hdlr,
						 int version);
XMLPUBFUN void XMLCALL
		xmlSAX2InitDefaultSAXHandler    (xmlSAXHandler *hdlr,
						 int warning);
#ifdef LIBXML_HTML_ENABLED
XMLPUBFUN void XMLCALL
		xmlSAX2InitHtmlDefaultSAXHandler(xmlSAXHandler *hdlr);
XMLPUBFUN void XMLCALL
		htmlDefaultSAXHandlerInit	(void);
#endif
#ifdef LIBXML_DOCB_ENABLED
XMLPUBFUN void XMLCALL
		xmlSAX2InitDocbDefaultSAXHandler(xmlSAXHandler *hdlr);
XMLPUBFUN void XMLCALL
		docbDefaultSAXHandlerInit	(void);
#endif
XMLPUBFUN void XMLCALL
		xmlDefaultSAXHandlerInit	(void);
#ifdef __cplusplus
}
#endif
#endif /* __XML_SAX2_H__ */
