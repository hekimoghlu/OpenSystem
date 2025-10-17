/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, June 10, 2023.
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
#ifndef __NANO_HTTP_H__
#define __NANO_HTTP_H__

#include <libxml/xmlversion.h>

#ifdef LIBXML_HTTP_ENABLED

#ifdef __cplusplus
extern "C" {
#endif
XMLPUBFUN void XMLCALL
	xmlNanoHTTPInit		(void);
XMLPUBFUN void XMLCALL
	xmlNanoHTTPCleanup	(void);
XMLPUBFUN void XMLCALL
	xmlNanoHTTPScanProxy	(const char *URL);
XMLPUBFUN int XMLCALL
	xmlNanoHTTPFetch	(const char *URL,
				 const char *filename,
				 char **contentType);
XMLPUBFUN void * XMLCALL
	xmlNanoHTTPMethod	(const char *URL,
				 const char *method,
				 const char *input,
				 char **contentType,
				 const char *headers,
				 int   ilen);
XMLPUBFUN void * XMLCALL
	xmlNanoHTTPMethodRedir	(const char *URL,
				 const char *method,
				 const char *input,
				 char **contentType,
				 char **redir,
				 const char *headers,
				 int   ilen);
XMLPUBFUN void * XMLCALL
	xmlNanoHTTPOpen		(const char *URL,
				 char **contentType);
XMLPUBFUN void * XMLCALL
	xmlNanoHTTPOpenRedir	(const char *URL,
				 char **contentType,
				 char **redir);
XMLPUBFUN int XMLCALL
	xmlNanoHTTPReturnCode	(void *ctx);
XMLPUBFUN const char * XMLCALL
	xmlNanoHTTPAuthHeader	(void *ctx);
XMLPUBFUN const char * XMLCALL
	xmlNanoHTTPRedir	(void *ctx);
XMLPUBFUN int XMLCALL
	xmlNanoHTTPContentLength( void * ctx );
XMLPUBFUN const char * XMLCALL
	xmlNanoHTTPEncoding	(void *ctx);
XMLPUBFUN const char * XMLCALL
	xmlNanoHTTPMimeType	(void *ctx);
XMLPUBFUN int XMLCALL
	xmlNanoHTTPRead		(void *ctx,
				 void *dest,
				 int len);
#ifdef LIBXML_OUTPUT_ENABLED
XMLPUBFUN int XMLCALL
	xmlNanoHTTPSave		(void *ctxt,
				 const char *filename);
#endif /* LIBXML_OUTPUT_ENABLED */
XMLPUBFUN void XMLCALL
	xmlNanoHTTPClose	(void *ctx);
#ifdef __cplusplus
}
#endif

#endif /* LIBXML_HTTP_ENABLED */
#endif /* __NANO_HTTP_H__ */
