/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 6, 2022.
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
#ifndef __XML_C14N_H__
#define __XML_C14N_H__

#include <libxml/xmlversion.h>

#ifdef LIBXML_C14N_ENABLED
#ifdef LIBXML_OUTPUT_ENABLED

#include <libxml/tree.h>
#include <libxml/xpath.h>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/*
 * XML Canonicalization
 * http://www.w3.org/TR/xml-c14n
 *
 * Exclusive XML Canonicalization
 * http://www.w3.org/TR/xml-exc-c14n
 *
 * Canonical form of an XML document could be created if and only if
 *  a) default attributes (if any) are added to all nodes
 *  b) all character and parsed entity references are resolved
 * In order to achieve this in libxml2 the document MUST be loaded with
 * following global settings:
 *
 *    xmlLoadExtDtdDefaultValue = XML_DETECT_IDS | XML_COMPLETE_ATTRS;
 *    xmlSubstituteEntitiesDefault(1);
 *
 * or corresponding parser context setting:
 *    xmlParserCtxtPtr ctxt;
 *
 *    ...
 *    ctxt->loadsubset = XML_DETECT_IDS | XML_COMPLETE_ATTRS;
 *    ctxt->replaceEntities = 1;
 *    ...
 */

/*
 * xmlC14NMode:
 *
 * Predefined values for C14N modes
 *
 */
typedef enum {
    XML_C14N_1_0            = 0,    /* Original C14N 1.0 spec */
    XML_C14N_EXCLUSIVE_1_0  = 1,    /* Exclusive C14N 1.0 spec */
    XML_C14N_1_1            = 2     /* C14N 1.1 spec */
} xmlC14NMode;

XMLPUBFUN int XMLCALL
		xmlC14NDocSaveTo	(xmlDocPtr doc,
					 xmlNodeSetPtr nodes,
					 int mode, /* a xmlC14NMode */
					 xmlChar **inclusive_ns_prefixes,
					 int with_comments,
					 xmlOutputBufferPtr buf);

XMLPUBFUN int XMLCALL
		xmlC14NDocDumpMemory	(xmlDocPtr doc,
					 xmlNodeSetPtr nodes,
					 int mode, /* a xmlC14NMode */
					 xmlChar **inclusive_ns_prefixes,
					 int with_comments,
					 xmlChar **doc_txt_ptr);

XMLPUBFUN int XMLCALL
		xmlC14NDocSave		(xmlDocPtr doc,
					 xmlNodeSetPtr nodes,
					 int mode, /* a xmlC14NMode */
					 xmlChar **inclusive_ns_prefixes,
					 int with_comments,
					 const char* filename,
					 int compression);


/**
 * This is the core C14N function
 */
/**
 * xmlC14NIsVisibleCallback:
 * @user_data: user data
 * @node: the current node
 * @parent: the parent node
 *
 * Signature for a C14N callback on visible nodes
 *
 * Returns 1 if the node should be included
 */
typedef int (*xmlC14NIsVisibleCallback)	(void* user_data,
					 xmlNodePtr node,
					 xmlNodePtr parent);

XMLPUBFUN int XMLCALL
		xmlC14NExecute		(xmlDocPtr doc,
					 xmlC14NIsVisibleCallback is_visible_callback,
					 void* user_data,
					 int mode, /* a xmlC14NMode */
					 xmlChar **inclusive_ns_prefixes,
					 int with_comments,
					 xmlOutputBufferPtr buf);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* LIBXML_OUTPUT_ENABLED */
#endif /* LIBXML_C14N_ENABLED */
#endif /* __XML_C14N_H__ */

