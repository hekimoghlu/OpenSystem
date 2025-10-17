/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 3, 2023.
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
#ifndef __XML_URI_H__
#define __XML_URI_H__

#include <libxml/xmlversion.h>
#include <libxml/tree.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * xmlURI:
 *
 * A parsed URI reference. This is a struct containing the various fields
 * as described in RFC 2396 but separated for further processing.
 *
 * Note: query is a deprecated field which is incorrectly unescaped.
 * query_raw takes precedence over query if the former is set.
 * See: http://mail.gnome.org/archives/xml/2007-April/thread.html#00127
 */
typedef struct _xmlURI xmlURI;
typedef xmlURI *xmlURIPtr;
struct _xmlURI {
    char *scheme;	/* the URI scheme */
    char *opaque;	/* opaque part */
    char *authority;	/* the authority part */
    char *server;	/* the server part */
    char *user;		/* the user part */
    int port;		/* the port number */
    char *path;		/* the path string */
    char *query;	/* the query string (deprecated - use with caution) */
    char *fragment;	/* the fragment identifier */
    int  cleanup;	/* parsing potentially unclean URI */
    char *query_raw;	/* the query string (as it appears in the URI) */
};

/*
 * This function is in tree.h:
 * xmlChar *	xmlNodeGetBase	(xmlDocPtr doc,
 *                               xmlNodePtr cur);
 */
XMLPUBFUN xmlURIPtr XMLCALL
		xmlCreateURI		(void);
XMLPUBFUN xmlChar * XMLCALL
		xmlBuildURI		(const xmlChar *URI,
					 const xmlChar *base);
XMLPUBFUN xmlChar * XMLCALL
		xmlBuildRelativeURI	(const xmlChar *URI,
					 const xmlChar *base);
XMLPUBFUN xmlURIPtr XMLCALL
		xmlParseURI		(const char *str);
XMLPUBFUN xmlURIPtr XMLCALL
		xmlParseURIRaw		(const char *str,
					 int raw);
XMLPUBFUN int XMLCALL
		xmlParseURIReference	(xmlURIPtr uri,
					 const char *str);
XMLPUBFUN xmlChar * XMLCALL
		xmlSaveUri		(xmlURIPtr uri);
XMLPUBFUN void XMLCALL
		xmlPrintURI		(FILE *stream,
					 xmlURIPtr uri);
XMLPUBFUN xmlChar * XMLCALL
		xmlURIEscapeStr         (const xmlChar *str,
					 const xmlChar *list);
XMLPUBFUN char * XMLCALL
		xmlURIUnescapeString	(const char *str,
					 int len,
					 char *target);
XMLPUBFUN int XMLCALL
		xmlNormalizeURIPath	(char *path);
XMLPUBFUN xmlChar * XMLCALL
		xmlURIEscape		(const xmlChar *str);
XMLPUBFUN void XMLCALL
		xmlFreeURI		(xmlURIPtr uri);
XMLPUBFUN xmlChar* XMLCALL
		xmlCanonicPath		(const xmlChar *path);
XMLPUBFUN xmlChar* XMLCALL
		xmlPathToURI		(const xmlChar *path);

#ifdef __cplusplus
}
#endif
#endif /* __XML_URI_H__ */
