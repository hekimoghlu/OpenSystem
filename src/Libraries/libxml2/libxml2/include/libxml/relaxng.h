/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, April 28, 2024.
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
#ifndef __XML_RELAX_NG__
#define __XML_RELAX_NG__

#include <libxml/xmlversion.h>
#include <libxml/hash.h>
#include <libxml/xmlstring.h>

#ifdef LIBXML_SCHEMAS_ENABLED

#ifdef __cplusplus
extern "C" {
#endif

typedef struct _xmlRelaxNG xmlRelaxNG;
typedef xmlRelaxNG *xmlRelaxNGPtr;


/**
 * xmlRelaxNGValidityErrorFunc:
 * @ctx: the validation context
 * @msg: the message
 * @...: extra arguments
 *
 * Signature of an error callback from a Relax-NG validation
 */
typedef void (XMLCDECL *xmlRelaxNGValidityErrorFunc) (void *ctx,
						      const char *msg,
						      ...) LIBXML_ATTR_FORMAT(2,3);

/**
 * xmlRelaxNGValidityWarningFunc:
 * @ctx: the validation context
 * @msg: the message
 * @...: extra arguments
 *
 * Signature of a warning callback from a Relax-NG validation
 */
typedef void (XMLCDECL *xmlRelaxNGValidityWarningFunc) (void *ctx,
							const char *msg,
							...) LIBXML_ATTR_FORMAT(2,3);

/**
 * A schemas validation context
 */
typedef struct _xmlRelaxNGParserCtxt xmlRelaxNGParserCtxt;
typedef xmlRelaxNGParserCtxt *xmlRelaxNGParserCtxtPtr;

typedef struct _xmlRelaxNGValidCtxt xmlRelaxNGValidCtxt;
typedef xmlRelaxNGValidCtxt *xmlRelaxNGValidCtxtPtr;

/*
 * xmlRelaxNGValidErr:
 *
 * List of possible Relax NG validation errors
 */
typedef enum {
    XML_RELAXNG_OK = 0,
    XML_RELAXNG_ERR_MEMORY,
    XML_RELAXNG_ERR_TYPE,
    XML_RELAXNG_ERR_TYPEVAL,
    XML_RELAXNG_ERR_DUPID,
    XML_RELAXNG_ERR_TYPECMP,
    XML_RELAXNG_ERR_NOSTATE,
    XML_RELAXNG_ERR_NODEFINE,
    XML_RELAXNG_ERR_LISTEXTRA,
    XML_RELAXNG_ERR_LISTEMPTY,
    XML_RELAXNG_ERR_INTERNODATA,
    XML_RELAXNG_ERR_INTERSEQ,
    XML_RELAXNG_ERR_INTEREXTRA,
    XML_RELAXNG_ERR_ELEMNAME,
    XML_RELAXNG_ERR_ATTRNAME,
    XML_RELAXNG_ERR_ELEMNONS,
    XML_RELAXNG_ERR_ATTRNONS,
    XML_RELAXNG_ERR_ELEMWRONGNS,
    XML_RELAXNG_ERR_ATTRWRONGNS,
    XML_RELAXNG_ERR_ELEMEXTRANS,
    XML_RELAXNG_ERR_ATTREXTRANS,
    XML_RELAXNG_ERR_ELEMNOTEMPTY,
    XML_RELAXNG_ERR_NOELEM,
    XML_RELAXNG_ERR_NOTELEM,
    XML_RELAXNG_ERR_ATTRVALID,
    XML_RELAXNG_ERR_CONTENTVALID,
    XML_RELAXNG_ERR_EXTRACONTENT,
    XML_RELAXNG_ERR_INVALIDATTR,
    XML_RELAXNG_ERR_DATAELEM,
    XML_RELAXNG_ERR_VALELEM,
    XML_RELAXNG_ERR_LISTELEM,
    XML_RELAXNG_ERR_DATATYPE,
    XML_RELAXNG_ERR_VALUE,
    XML_RELAXNG_ERR_LIST,
    XML_RELAXNG_ERR_NOGRAMMAR,
    XML_RELAXNG_ERR_EXTRADATA,
    XML_RELAXNG_ERR_LACKDATA,
    XML_RELAXNG_ERR_INTERNAL,
    XML_RELAXNG_ERR_ELEMWRONG,
    XML_RELAXNG_ERR_TEXTWRONG
} xmlRelaxNGValidErr;

/*
 * xmlRelaxNGParserFlags:
 *
 * List of possible Relax NG Parser flags
 */
typedef enum {
    XML_RELAXNGP_NONE = 0,
    XML_RELAXNGP_FREE_DOC = 1,
    XML_RELAXNGP_CRNG = 2
} xmlRelaxNGParserFlag;

XMLPUBFUN int XMLCALL
		    xmlRelaxNGInitTypes		(void);
XMLPUBFUN void XMLCALL
		    xmlRelaxNGCleanupTypes	(void);

/*
 * Interfaces for parsing.
 */
XMLPUBFUN xmlRelaxNGParserCtxtPtr XMLCALL
		    xmlRelaxNGNewParserCtxt	(const char *URL);
XMLPUBFUN xmlRelaxNGParserCtxtPtr XMLCALL
		    xmlRelaxNGNewMemParserCtxt	(const char *buffer,
						 int size);
XMLPUBFUN xmlRelaxNGParserCtxtPtr XMLCALL
		    xmlRelaxNGNewDocParserCtxt	(xmlDocPtr doc);

XMLPUBFUN int XMLCALL
		    xmlRelaxParserSetFlag	(xmlRelaxNGParserCtxtPtr ctxt,
						 int flag);

XMLPUBFUN void XMLCALL
		    xmlRelaxNGFreeParserCtxt	(xmlRelaxNGParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
		    xmlRelaxNGSetParserErrors(xmlRelaxNGParserCtxtPtr ctxt,
					 xmlRelaxNGValidityErrorFunc err,
					 xmlRelaxNGValidityWarningFunc warn,
					 void *ctx);
XMLPUBFUN int XMLCALL
		    xmlRelaxNGGetParserErrors(xmlRelaxNGParserCtxtPtr ctxt,
					 xmlRelaxNGValidityErrorFunc *err,
					 xmlRelaxNGValidityWarningFunc *warn,
					 void **ctx);
XMLPUBFUN void XMLCALL
		    xmlRelaxNGSetParserStructuredErrors(
					 xmlRelaxNGParserCtxtPtr ctxt,
					 xmlStructuredErrorFunc serror,
					 void *ctx);
XMLPUBFUN xmlRelaxNGPtr XMLCALL
		    xmlRelaxNGParse		(xmlRelaxNGParserCtxtPtr ctxt);
XMLPUBFUN void XMLCALL
		    xmlRelaxNGFree		(xmlRelaxNGPtr schema);
#ifdef LIBXML_OUTPUT_ENABLED
XMLPUBFUN void XMLCALL
		    xmlRelaxNGDump		(FILE *output,
					 xmlRelaxNGPtr schema);
XMLPUBFUN void XMLCALL
		    xmlRelaxNGDumpTree	(FILE * output,
					 xmlRelaxNGPtr schema);
#endif /* LIBXML_OUTPUT_ENABLED */
/*
 * Interfaces for validating
 */
XMLPUBFUN void XMLCALL
		    xmlRelaxNGSetValidErrors(xmlRelaxNGValidCtxtPtr ctxt,
					 xmlRelaxNGValidityErrorFunc err,
					 xmlRelaxNGValidityWarningFunc warn,
					 void *ctx);
XMLPUBFUN int XMLCALL
		    xmlRelaxNGGetValidErrors(xmlRelaxNGValidCtxtPtr ctxt,
					 xmlRelaxNGValidityErrorFunc *err,
					 xmlRelaxNGValidityWarningFunc *warn,
					 void **ctx);
XMLPUBFUN void XMLCALL
			xmlRelaxNGSetValidStructuredErrors(xmlRelaxNGValidCtxtPtr ctxt,
					  xmlStructuredErrorFunc serror, void *ctx);
XMLPUBFUN xmlRelaxNGValidCtxtPtr XMLCALL
		    xmlRelaxNGNewValidCtxt	(xmlRelaxNGPtr schema);
XMLPUBFUN void XMLCALL
		    xmlRelaxNGFreeValidCtxt	(xmlRelaxNGValidCtxtPtr ctxt);
XMLPUBFUN int XMLCALL
		    xmlRelaxNGValidateDoc	(xmlRelaxNGValidCtxtPtr ctxt,
						 xmlDocPtr doc);
/*
 * Interfaces for progressive validation when possible
 */
XMLPUBFUN int XMLCALL
		    xmlRelaxNGValidatePushElement	(xmlRelaxNGValidCtxtPtr ctxt,
					 xmlDocPtr doc,
					 xmlNodePtr elem);
XMLPUBFUN int XMLCALL
		    xmlRelaxNGValidatePushCData	(xmlRelaxNGValidCtxtPtr ctxt,
					 const xmlChar *data,
					 int len);
XMLPUBFUN int XMLCALL
		    xmlRelaxNGValidatePopElement	(xmlRelaxNGValidCtxtPtr ctxt,
					 xmlDocPtr doc,
					 xmlNodePtr elem);
XMLPUBFUN int XMLCALL
		    xmlRelaxNGValidateFullElement	(xmlRelaxNGValidCtxtPtr ctxt,
					 xmlDocPtr doc,
					 xmlNodePtr elem);

#ifdef __cplusplus
}
#endif

#endif /* LIBXML_SCHEMAS_ENABLED */

#endif /* __XML_RELAX_NG__ */
