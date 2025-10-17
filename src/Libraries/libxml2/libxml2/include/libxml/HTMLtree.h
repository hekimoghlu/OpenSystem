/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, June 12, 2025.
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
#ifndef __HTML_TREE_H__
#define __HTML_TREE_H__

#include <stdio.h>
#include <libxml/xmlversion.h>
#include <libxml/tree.h>
#include <libxml/HTMLparser.h>

#ifdef LIBXML_HTML_ENABLED

#ifdef __cplusplus
extern "C" {
#endif


/**
 * HTML_TEXT_NODE:
 *
 * Macro. A text node in a HTML document is really implemented
 * the same way as a text node in an XML document.
 */
#define HTML_TEXT_NODE		XML_TEXT_NODE
/**
 * HTML_ENTITY_REF_NODE:
 *
 * Macro. An entity reference in a HTML document is really implemented
 * the same way as an entity reference in an XML document.
 */
#define HTML_ENTITY_REF_NODE	XML_ENTITY_REF_NODE
/**
 * HTML_COMMENT_NODE:
 *
 * Macro. A comment in a HTML document is really implemented
 * the same way as a comment in an XML document.
 */
#define HTML_COMMENT_NODE	XML_COMMENT_NODE
/**
 * HTML_PRESERVE_NODE:
 *
 * Macro. A preserved node in a HTML document is really implemented
 * the same way as a CDATA section in an XML document.
 */
#define HTML_PRESERVE_NODE	XML_CDATA_SECTION_NODE
/**
 * HTML_PI_NODE:
 *
 * Macro. A processing instruction in a HTML document is really implemented
 * the same way as a processing instruction in an XML document.
 */
#define HTML_PI_NODE		XML_PI_NODE

XMLPUBFUN htmlDocPtr XMLCALL
		htmlNewDoc		(const xmlChar *URI,
					 const xmlChar *ExternalID);
XMLPUBFUN htmlDocPtr XMLCALL
		htmlNewDocNoDtD		(const xmlChar *URI,
					 const xmlChar *ExternalID);
XMLPUBFUN const xmlChar * XMLCALL
		htmlGetMetaEncoding	(htmlDocPtr doc);
XMLPUBFUN int XMLCALL
		htmlSetMetaEncoding	(htmlDocPtr doc,
					 const xmlChar *encoding);
#ifdef LIBXML_OUTPUT_ENABLED
XMLPUBFUN void XMLCALL
		htmlDocDumpMemory	(xmlDocPtr cur,
					 xmlChar **mem,
					 int *size);
XMLPUBFUN void XMLCALL
		htmlDocDumpMemoryFormat	(xmlDocPtr cur,
					 xmlChar **mem,
					 int *size,
					 int format);
XMLPUBFUN int XMLCALL
		htmlDocDump		(FILE *f,
					 xmlDocPtr cur);
XMLPUBFUN int XMLCALL
		htmlSaveFile		(const char *filename,
					 xmlDocPtr cur);
XMLPUBFUN int XMLCALL
		htmlNodeDump		(xmlBufferPtr buf,
					 xmlDocPtr doc,
					 xmlNodePtr cur);
XMLPUBFUN void XMLCALL
		htmlNodeDumpFile	(FILE *out,
					 xmlDocPtr doc,
					 xmlNodePtr cur);
XMLPUBFUN int XMLCALL
		htmlNodeDumpFileFormat	(FILE *out,
					 xmlDocPtr doc,
					 xmlNodePtr cur,
					 const char *encoding,
					 int format);
XMLPUBFUN int XMLCALL
		htmlSaveFileEnc		(const char *filename,
					 xmlDocPtr cur,
					 const char *encoding);
XMLPUBFUN int XMLCALL
		htmlSaveFileFormat	(const char *filename,
					 xmlDocPtr cur,
					 const char *encoding,
					 int format);

XMLPUBFUN void XMLCALL
		htmlNodeDumpFormatOutput(xmlOutputBufferPtr buf,
					 xmlDocPtr doc,
					 xmlNodePtr cur,
					 const char *encoding,
					 int format);
XMLPUBFUN void XMLCALL
		htmlDocContentDumpOutput(xmlOutputBufferPtr buf,
					 xmlDocPtr cur,
					 const char *encoding);
XMLPUBFUN void XMLCALL
		htmlDocContentDumpFormatOutput(xmlOutputBufferPtr buf,
					 xmlDocPtr cur,
					 const char *encoding,
					 int format);
XMLPUBFUN void XMLCALL
		htmlNodeDumpOutput	(xmlOutputBufferPtr buf,
					 xmlDocPtr doc,
					 xmlNodePtr cur,
					 const char *encoding);

#endif /* LIBXML_OUTPUT_ENABLED */

XMLPUBFUN int XMLCALL
		htmlIsBooleanAttr	(const xmlChar *name);


#ifdef __cplusplus
}
#endif

#endif /* LIBXML_HTML_ENABLED */

#endif /* __HTML_TREE_H__ */

